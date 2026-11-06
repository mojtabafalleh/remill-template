#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "exepath.hpp"

#include <glog/logging.h>

#include <remill/Arch/Arch.h>
#include <remill/Arch/Instruction.h>
#include <remill/Arch/Name.h>
#include <remill/BC/ABI.h>
#include <remill/BC/IntrinsicTable.h>
#include <remill/BC/Lifter.h>
#include <remill/BC/Optimizer.h>
#include <remill/BC/Util.h>
#include <remill/BC/Version.h>
#include <remill/OS/OS.h>
#include <remill/Version/Version.h>

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/IR/LegacyPassManager.h>

#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/DeadStoreElimination.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/MemCpyOptimizer.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Scalar/BDCE.h>
#include <llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h>

#include <LIEF/LIEF.hpp>

#include <capstone/capstone.h>

static constexpr bool kKeepRamGlobal = true;
static constexpr bool kLowerRamAtEnd = true;
static constexpr bool kSSAState = false;
static constexpr bool kAssumeNoRedZone = true;
static constexpr int kSimplifyRounds = 8;

static size_t CountInstructions(const llvm::Function &function) {

  size_t count = 0;

  for (const auto &bb : function) {
    count += bb.size();
  }

  return count;
}

void aggressiveSimplify(llvm::Function *function) {

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassBuilder pb;
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::FunctionPassManager fpm = pb.buildFunctionSimplificationPipeline(
      llvm::OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::None);

  fpm.addPass(llvm::SCCPPass());
  fpm.addPass(llvm::AggressiveInstCombinePass());
  fpm.addPass(llvm::InstCombinePass());
  fpm.addPass(llvm::GVNPass());
  fpm.addPass(llvm::MemCpyOptPass());
  fpm.addPass(llvm::DSEPass());
  fpm.addPass(llvm::BDCEPass());
  fpm.addPass(llvm::ADCEPass());
  fpm.addPass(llvm::SimplifyCFGPass());
  fpm.addPass(llvm::InstCombinePass());

  size_t previous = CountInstructions(*function);

  for (int i = 0; i < kSimplifyRounds; ++i) {

    fpm.run(*function, fam);

    size_t current = CountInstructions(*function);

    if (current >= previous) {
      break;
    }

    previous = current;
  }
}

static std::optional<uint64_t> RegOffset(
    const remill::Arch *arch,
    const char *upper,
    const char *lower) {

  if (const auto *reg = arch->RegisterByName(upper)) {
    return reg->offset;
  }

  if (const auto *reg = arch->RegisterByName(lower)) {
    return reg->offset;
  }

  return std::nullopt;
}

static bool StateOffset(
    llvm::Value *ptr,
    llvm::Value *state,
    const llvm::DataLayout &dl,
    int64_t &off) {

  llvm::APInt acc(dl.getIndexSizeInBits(0), 0);

  const llvm::Value *base =
      ptr->stripAndAccumulateConstantOffsets(dl, acc, true);

  if (base != state) {
    return false;
  }

  off = acc.getSExtValue();

  return true;
}

static void PeelConstAdd(
    llvm::Value *v,
    llvm::Value *&sym,
    int64_t &off) {

  sym = v;
  off = 0;

  for (;;) {

    auto *bo = llvm::dyn_cast<llvm::BinaryOperator>(sym);

    if (!bo || bo->getOpcode() != llvm::Instruction::Add) {
      return;
    }

    if (auto *c = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(1))) {
      off += c->getSExtValue();
      sym = bo->getOperand(0);
      continue;
    }

    if (auto *c = llvm::dyn_cast<llvm::ConstantInt>(bo->getOperand(0))) {
      off += c->getSExtValue();
      sym = bo->getOperand(1);
      continue;
    }

    return;
  }
}

static bool DecomposeRamPtr(
    llvm::Value *ptr,
    llvm::GlobalVariable *ram,
    const llvm::DataLayout &dl,
    llvm::Value *&sym,
    int64_t &off) {

  llvm::APInt acc(dl.getIndexSizeInBits(0), 0);

  llvm::Value *base =
      const_cast<llvm::Value *>(
          ptr->stripAndAccumulateConstantOffsets(dl, acc, true));

  int64_t constOff = acc.getSExtValue();

  if (base == ram) {
    sym = nullptr;
    off = constOff;
    return true;
  }

  auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(base);

  if (!gep ||
      gep->getPointerOperand() != ram ||
      gep->getNumIndices() != 1) {
    return false;
  }

  int64_t peeled = 0;

  PeelConstAdd(gep->getOperand(1), sym, peeled);

  off = constOff + peeled;

  return true;
}

namespace ramclean {

enum class Mode {
  KeepGlobal,
  IntToPtr,
};

static llvm::Value *GuestAddr(
    llvm::GEPOperator *gep, llvm::GlobalVariable *ram) {

  if (gep->getPointerOperand() != ram) {
    return nullptr;
  }

  switch (gep->getNumIndices()) {

    case 1:
      return gep->getOperand(1);

    case 2: {
      auto *first = llvm::dyn_cast<llvm::ConstantInt>(gep->getOperand(1));
      return (first && first->isZero()) ? gep->getOperand(2) : nullptr;
    }

    default:
      return nullptr;
  }
}

static bool Rewrite(llvm::Module &module, Mode mode) {

  auto *ram = module.getGlobalVariable("RAM");

  if (!ram) {
    return false;
  }

  auto &ctx = module.getContext();

  auto *i8 = llvm::Type::getInt8Ty(ctx);
  auto *i64 = llvm::Type::getInt64Ty(ctx);
  auto *ptrTy = llvm::PointerType::get(ctx, ram->getAddressSpace());

  std::set<llvm::User *> users(ram->user_begin(), ram->user_end());

  std::vector<llvm::Instruction *> deadInsts;
  std::vector<llvm::Constant *> deadConsts;

  unsigned rewritten = 0;
  unsigned skipped = 0;

  for (auto *user : users) {

    auto *gep = llvm::dyn_cast<llvm::GEPOperator>(user);

    llvm::Value *addr = gep ? GuestAddr(gep, ram) : nullptr;

    if (!addr) {
      ++skipped;
      continue;
    }

    if (mode == Mode::KeepGlobal &&
        gep->getNumIndices() == 1 &&
        gep->getSourceElementType()->isIntegerTy(8)) {
      continue;
    }

    if (auto *inst = llvm::dyn_cast<llvm::GetElementPtrInst>(user)) {

      llvm::IRBuilder<> ir(inst);

      llvm::Value *idx = ir.CreateZExtOrTrunc(addr, i64);

      llvm::Value *repl =
          (mode == Mode::KeepGlobal)
              ? ir.CreateGEP(i8, ram, {idx})
              : ir.CreateIntToPtr(idx, ptrTy);

      inst->replaceAllUsesWith(repl);
      deadInsts.push_back(inst);

    } else if (auto *ce = llvm::dyn_cast<llvm::ConstantExpr>(user)) {

      auto *cidx = llvm::dyn_cast<llvm::Constant>(addr);

      if (!cidx) {
        ++skipped;
        continue;
      }

      llvm::Constant *idx = nullptr;

      if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(cidx)) {
        idx = llvm::ConstantInt::get(i64, ci->getValue().zextOrTrunc(64));
      } else if (cidx->getType() == i64) {
        idx = cidx;
      } else {
        ++skipped;
        continue;
      }

      llvm::Constant *repl =
          (mode == Mode::KeepGlobal)
              ? llvm::ConstantExpr::getGetElementPtr(i8, ram, idx)
              : llvm::ConstantExpr::getIntToPtr(idx, ptrTy);

      ce->replaceAllUsesWith(repl);
      deadConsts.push_back(ce);

    } else {
      ++skipped;
      continue;
    }

    ++rewritten;
  }

  for (auto *inst : deadInsts) {
    inst->eraseFromParent();
  }

  for (auto *c : deadConsts) {
    c->destroyConstant();
  }

  if (mode == Mode::IntToPtr) {

    if (!ram->use_empty()) {
      ram->replaceAllUsesWith(llvm::ConstantPointerNull::get(ptrTy));
    }

    ram->eraseFromParent();

    for (auto &f : module) {
      if (!f.isDeclaration()) {
        f.addFnAttr(llvm::Attribute::NullPointerIsValid);
      }
    }
  }

  llvm::outs() << "[RAM] "
               << (mode == Mode::KeepGlobal ? "gep-i8" : "inttoptr")
               << ": rewritten " << rewritten
               << ", skipped " << skipped << "\n";

  return true;
}

}  // namespace ramclean

static void AnnotateFlagRanges(
    const remill::Arch *arch,
    llvm::Function &f) {

  std::set<int64_t> flagOffsets;

  static const char *kFlags[][2] = {
      {"AF", "af"}, {"CF", "cf"}, {"DF", "df"}, {"OF", "of"},
      {"PF", "pf"}, {"SF", "sf"}, {"ZF", "zf"}};

  for (const auto &names : kFlags) {
    if (auto off = RegOffset(arch, names[0], names[1])) {
      flagOffsets.insert(static_cast<int64_t>(*off));
    } else {
      llvm::errs() << "Warning: flag register " << names[0]
                   << " not found in arch\n";
    }
  }

  if (flagOffsets.empty()) {
    return;
  }

  llvm::MDBuilder mdb(f.getContext());

  auto *range = mdb.createRange(llvm::APInt(8, 0), llvm::APInt(8, 2));

  const auto &dl = f.getParent()->getDataLayout();

  llvm::Value *state = f.getArg(0);

  unsigned annotated = 0;

  for (auto &bb : f) {
    for (auto &inst : bb) {

      auto *ld = llvm::dyn_cast<llvm::LoadInst>(&inst);

      if (!ld || !ld->getType()->isIntegerTy(8)) {
        continue;
      }

      int64_t off = 0;

      if (StateOffset(ld->getPointerOperand(), state, dl, off) &&
          flagOffsets.count(off)) {

        ld->setMetadata(llvm::LLVMContext::MD_range, range);
        ++annotated;
      }
    }
  }

  llvm::outs() << "Annotated " << annotated << " flag load(s) with !range\n";
}

static bool RemoveDeadStackStores(
    const remill::Arch *arch,
    llvm::Function &f) {

  auto rspOff = RegOffset(arch, "RSP", "rsp");

  if (!rspOff) {
    llvm::errs() << "RSP register not found, skipping stack-store pass\n";
    return false;
  }

  auto *ram = f.getParent()->getGlobalVariable("RAM");

  if (!ram) {
    return false;
  }

  const auto &dl = f.getParent()->getDataLayout();

  llvm::Value *state = f.getArg(0);

  std::set<llvm::Value *> initialRsp;

  for (auto &bb : f) {
    for (auto &inst : bb) {

      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
        if (call->mayReadOrWriteMemory()) {
          llvm::errs() << "Call touching memory found, "
                          "skipping stack-store pass\n";
          return false;
        }
      }

      if (auto *ld = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
        int64_t off = 0;
        if (StateOffset(ld->getPointerOperand(), state, dl, off) &&
            off == static_cast<int64_t>(*rspOff) &&
            ld->getType()->isIntegerTy(64)) {
          initialRsp.insert(ld);
        }
      }
    }
  }

  if (initialRsp.empty()) {
    return false;
  }

  int64_t finalDelta = 0;
  bool seenRspStore = false;

  for (auto &bb : f) {
    for (auto &inst : bb) {

      auto *st = llvm::dyn_cast<llvm::StoreInst>(&inst);

      if (!st) {
        continue;
      }

      int64_t off = 0;

      if (!StateOffset(st->getPointerOperand(), state, dl, off) ||
          off != static_cast<int64_t>(*rspOff)) {
        continue;
      }

      llvm::Value *sym = nullptr;
      int64_t delta = 0;

      PeelConstAdd(st->getValueOperand(), sym, delta);

      if (!initialRsp.count(sym) || (seenRspStore && delta != finalDelta)) {
        llvm::errs() << "Inconsistent final RSP, "
                        "skipping stack-store pass\n";
        return false;
      }

      finalDelta = delta;
      seenRspStore = true;
    }
  }

  struct Access {
    llvm::Value *sym;
    int64_t off;
    int64_t size;
  };

  std::vector<Access> loads;
  std::vector<std::pair<llvm::StoreInst *, Access>> stores;

  for (auto &bb : f) {
    for (auto &inst : bb) {

      llvm::Value *ptr = nullptr;
      llvm::Type *ty = nullptr;
      bool isStore = false;

      if (auto *ld = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
        ptr = ld->getPointerOperand();
        ty = ld->getType();
      } else if (auto *st = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
        ptr = st->getPointerOperand();
        ty = st->getValueOperand()->getType();
        isStore = true;
      } else {
        continue;
      }

      const llvm::Value *under = llvm::getUnderlyingObject(ptr);

      if (under != ram) {
        continue;
      }

      llvm::Value *sym = nullptr;
      int64_t off = 0;

      bool ok = DecomposeRamPtr(ptr, ram, dl, sym, off);

      int64_t size =
          static_cast<int64_t>(dl.getTypeStoreSize(ty).getFixedValue());

      if (!ok) {
        if (!isStore) {
          llvm::errs() << "Undecomposable RAM load, "
                          "skipping stack-store pass\n";
          return false;
        }
        continue;
      }

      if (isStore) {
        stores.push_back(
            {llvm::cast<llvm::StoreInst>(&inst), {sym, off, size}});
      } else {
        loads.push_back({sym, off, size});
      }
    }
  }

  for (const auto &l : loads) {
    if (!l.sym || !initialRsp.count(l.sym)) {
      llvm::errs() << "RAM load with foreign base, "
                      "skipping stack-store pass\n";
      return false;
    }
  }

  std::vector<llvm::StoreInst *> dead;

  for (const auto &[st, a] : stores) {

    if (!a.sym || !initialRsp.count(a.sym)) {
      continue;
    }

    if (a.off + a.size > finalDelta) {
      continue;
    }

    bool overlapped = false;

    for (const auto &l : loads) {
      if (l.sym == a.sym &&
          l.off < a.off + a.size &&
          a.off < l.off + l.size) {
        overlapped = true;
        break;
      }
    }

    if (!overlapped) {
      dead.push_back(st);
    }
  }

  for (auto *st : dead) {
    st->eraseFromParent();
  }

  llvm::outs() << "Removed " << dead.size()
               << " dead store(s) below RSP\n";

  return !dead.empty();
}

class CapstoneDisassembler {
 public:
  CapstoneDisassembler() {
    cs_err err = cs_open(CS_ARCH_X86, CS_MODE_64, &handle_);

    if (err != CS_ERR_OK) {
      LOG(FATAL) << "Failed to initialize Capstone: " << cs_strerror(err);
    }

    if (cs_option(handle_, CS_OPT_SYNTAX, CS_OPT_SYNTAX_INTEL) != CS_ERR_OK) {
      LOG(FATAL) << "Failed to set Capstone Intel syntax";
    }

    cs_option(handle_, CS_OPT_DETAIL, CS_OPT_OFF);
  }

  ~CapstoneDisassembler() {
    if (handle_) {
      cs_close(&handle_);
    }
  }

  CapstoneDisassembler(const CapstoneDisassembler &) = delete;
  CapstoneDisassembler &operator=(const CapstoneDisassembler &) = delete;

  bool Print(uint64_t address, const uint8_t *bytes, size_t size) const {

    if (!bytes || size == 0) {
      llvm::outs() << "[0x" << llvm::format_hex_no_prefix(address, 16)
                   << "] <no bytes>\n";
      return false;
    }

    cs_insn *insn = nullptr;

    size_t count = cs_disasm(handle_, bytes, size, address, 1, &insn);

    if (count == 0 || !insn) {
      llvm::outs() << "[0x" << llvm::format_hex_no_prefix(address, 16)
                   << "] <Capstone decode failed>\n";
      return false;
    }

    llvm::outs() << "[0x" << llvm::format_hex_no_prefix(address, 16) << "] ";

    for (size_t i = 0; i < insn[0].size; ++i) {
      llvm::outs() << llvm::format_hex_no_prefix(
                          static_cast<unsigned>(insn[0].bytes[i]), 2)
                   << " ";
    }

    for (size_t i = insn[0].size; i < 10; ++i) {
      llvm::outs() << "   ";
    }

    llvm::outs() << "  " << insn[0].mnemonic;

    if (insn[0].op_str[0] != '\0') {
      llvm::outs() << " " << insn[0].op_str;
    }

    llvm::outs() << "\n";

    cs_free(insn, count);

    return true;
  }

  std::string Disassemble(
      uint64_t address, const uint8_t *bytes, size_t size) const {

    cs_insn *insn = nullptr;

    size_t count = cs_disasm(handle_, bytes, size, address, 1, &insn);

    if (count == 0 || !insn) {
      return "<decode failed>";
    }

    std::string result = insn[0].mnemonic;

    if (insn[0].op_str[0] != '\0') {
      result += " ";
      result += insn[0].op_str;
    }

    cs_free(insn, count);

    return result;
  }

 private:
  csh handle_{};
};

class PEImage {
 public:
  explicit PEImage(const std::string &path) {
    binary_ = LIEF::PE::Parser::parse(path);

    if (!binary_) {
      LOG(FATAL) << "Failed to parse PE file: " << path;
    }

    imageBase_ = binary_->optional_header().imagebase();

    for (const auto &section : binary_->sections()) {
      uint64_t va = imageBase_ + section.virtual_address();

      auto span = section.content();

      std::vector<uint8_t> content(span.begin(), span.end());

      if (content.empty()) {
        continue;
      }

      sections_.emplace_back(va, std::move(content));
    }
  }

  uint64_t ImageBase() const { return imageBase_; }

  uint64_t EntryPoint() const {
    return imageBase_ + binary_->optional_header().addressof_entrypoint();
  }

  size_t Read(uint64_t addr, uint8_t *out, size_t size) const {

    for (const auto &sec : sections_) {
      uint64_t start = sec.first;
      uint64_t end = start + sec.second.size();

      if (addr >= start && addr < end) {
        size_t avail = static_cast<size_t>(end - addr);
        size_t n = std::min(avail, size);

        std::memcpy(out, sec.second.data() + (addr - start), n);

        return n;
      }
    }

    return 0;
  }

 private:
  std::unique_ptr<LIEF::PE::Binary> binary_;

  uint64_t imageBase_ = 0;

  std::vector<std::pair<uint64_t, std::vector<uint8_t>>> sections_;
};

static const char *CategoryName(remill::Instruction::Category category) {

  using Category = remill::Instruction::Category;

  switch (category) {
    case Category::kCategoryNormal:               return "NORMAL";
    case Category::kCategoryNoOp:                 return "NO_OP";
    case Category::kCategoryDirectJump:           return "DIRECT_JUMP";
    case Category::kCategoryDirectFunctionCall:   return "DIRECT_CALL";
    case Category::kCategoryConditionalBranch:    return "CONDITIONAL_BRANCH";
    case Category::kCategoryIndirectJump:         return "INDIRECT_JUMP";
    case Category::kCategoryIndirectFunctionCall: return "INDIRECT_CALL";
    case Category::kCategoryFunctionReturn:       return "RETURN";
    default:                                      return "OTHER";
  }
}

static bool SaveLiftedFunctionIR(
    llvm::Function &function, const std::string &basePath) {

  std::error_code ec;

  llvm::raw_fd_ostream out(basePath + ".ll", ec, llvm::sys::fs::OF_None);

  if (ec) {
    llvm::errs() << "Failed to open " << basePath << ".ll: " << ec.message()
                 << "\n";
    return false;
  }

  function.print(out);

  llvm::outs() << "Saved lifted function IR to " << basePath << ".ll\n";
  llvm::outs() << "Verifying function...\n";

  if (llvm::verifyFunction(function, &llvm::outs())) {
    llvm::outs() << "Function verification failed!\n";
    return false;
  }

  return true;
}

static llvm::Function *ForceInlineLiftedBlocks(
    llvm::Module &module,
    llvm::Function *entryFunction,
    const std::vector<llvm::Function *> &allFunctions) {

  llvm::outs() << "\nForce-inlining " << allFunctions.size()
               << " lifted block(s) into entry function...\n";

  entryFunction->setLinkage(llvm::GlobalValue::ExternalLinkage);

  for (auto *function : allFunctions) {

    if (!function || function == entryFunction) {
      continue;
    }

    function->removeFnAttr(llvm::Attribute::NoInline);
    function->addFnAttr(llvm::Attribute::AlwaysInline);

    function->setLinkage(llvm::GlobalValue::InternalLinkage);
  }

  llvm::legacy::PassManager pm;

  pm.add(llvm::createAlwaysInlinerLegacyPass());

  pm.run(module);

  llvm::outs() << "Force-inlining complete.\n";

  bool erasedAny = true;

  while (erasedAny) {

    erasedAny = false;

    for (auto it = module.begin(); it != module.end();) {

      llvm::Function &f = *it;
      ++it;

      if (&f == entryFunction) {
        continue;
      }

      bool isInternal = f.hasInternalLinkage() || f.hasPrivateLinkage();

      if (isInternal && f.use_empty()) {
        f.eraseFromParent();
        erasedAny = true;
      }
    }
  }

  return entryFunction;
}

static llvm::Function *BuildMultiEntryWrapper(
    llvm::Module &module,
    const std::vector<llvm::Function *> &entryFunctions) {

  if (entryFunctions.empty()) {
    return nullptr;
  }

  if (entryFunctions.size() == 1) {
    return entryFunctions.front();
  }

  llvm::Function *first = entryFunctions.front();

  auto *wrapper = llvm::Function::Create(
      first->getFunctionType(),
      llvm::GlobalValue::ExternalLinkage,
      "lifted_entry_all",
      &module);

  auto *entryBlock =
      llvm::BasicBlock::Create(module.getContext(), "entry", wrapper);

  llvm::IRBuilder<> ir(entryBlock);

  llvm::Value *state = wrapper->getArg(0);
  llvm::Value *pc = wrapper->getArg(1);
  llvm::Value *memory = wrapper->getArg(2);

  for (auto *fn : entryFunctions) {
    memory = ir.CreateCall(fn, {state, pc, memory});
  }

  ir.CreateRet(memory);

  llvm::outs() << "Created synthetic wrapper '" << wrapper->getName()
               << "' calling " << entryFunctions.size()
               << " independent lift entries.\n";

  return wrapper;
}

static llvm::Function *BuildSSAWrapper(
    const remill::Arch *arch, llvm::Module &module, llvm::Function *entry) {

  auto &ctx = module.getContext();

  llvm::Type *stateTy = arch->StateStructType();

  uint64_t stateSize = module.getDataLayout().getTypeAllocSize(stateTy);

  auto *wrapper = llvm::Function::Create(
      entry->getFunctionType(),
      llvm::GlobalValue::ExternalLinkage,
      "lifted_ssa",
      &module);

  wrapper->getArg(0)->setName("state");
  wrapper->getArg(1)->setName("program_counter");
  wrapper->getArg(2)->setName("memory");

  auto *bb = llvm::BasicBlock::Create(ctx, "entry", wrapper);

  llvm::IRBuilder<> ir(bb);

  auto *local = ir.CreateAlloca(stateTy, nullptr, "local_state");

  ir.CreateMemCpy(
      local, local->getAlign(),
      wrapper->getArg(0), llvm::MaybeAlign(1),
      stateSize);

  auto *mem = ir.CreateCall(
      entry, {local, wrapper->getArg(1), wrapper->getArg(2)});

  ir.CreateMemCpy(
      wrapper->getArg(0), llvm::MaybeAlign(1),
      local, local->getAlign(),
      stateSize);

  ir.CreateRet(mem);

  llvm::outs() << "Created SSA wrapper '" << wrapper->getName()
               << "' (State size = " << stateSize << " bytes)\n";

  return wrapper;
}

static bool hotpatchRemill(
    llvm::Module &module, const std::string &hotpatchPath) {

  if (!std::filesystem::exists(hotpatchPath)) {
    llvm::errs() << "Hotpatch file not found: " << hotpatchPath << "\n";
    return false;
  }

  llvm::SMDiagnostic error;

  auto patchModule =
      llvm::parseIRFile(hotpatchPath, error, module.getContext());

  if (!patchModule) {
    llvm::errs() << "Failed to parse hotpatch module: " << error.getMessage()
                 << "\n";
    return false;
  }

  patchModule->setDataLayout(module.getDataLayout());
  patchModule->setTargetTriple(module.getTargetTriple());

  for (const auto &global : patchModule->globals()) {

    const auto &globalName = global.getName().str();

    if (globalName.rfind("ISEL_", 0) == 0) {

      auto *existingGlobal = module.getGlobalVariable(globalName);

      if (existingGlobal) {

        existingGlobal->setName(globalName + "_original");

        llvm::outs() << "Hotpatching: " << globalName << "\n";
      }
    }
  }

  if (llvm::Linker::linkModules(
          module,
          std::move(patchModule),
          llvm::Linker::Flags::OverrideFromSrc)) {

    llvm::errs() << "Failed to link hotpatch module\n";
    return false;
  }

  return true;
}

struct LiftRange {
  uint64_t start;
  uint64_t end;
};

class RangeLifter {
 public:

  RangeLifter(
      const remill::Arch *arch,
      llvm::Module &module,
      const remill::IntrinsicTable &intrinsics,
      const PEImage &image,
      std::vector<LiftRange> ranges)
      : arch_(arch),
        module_(module),
        intrinsics_(intrinsics),
        image_(image),
        ranges_(std::move(ranges)) {}

  std::vector<llvm::Function *> Run() {

    std::vector<llvm::Function *> entryFunctions;
    entryFunctions.reserve(ranges_.size());

    for (const auto &range : ranges_) {
      entryFunctions.push_back(GetOrDeclareFunction(range.start));
      worklist_.push(range.start);
    }

    while (!worklist_.empty()) {

      uint64_t addr = worklist_.front();

      worklist_.pop();

      if (visited_.count(addr)) {
        continue;
      }

      visited_.insert(addr);

      llvm::outs() << "\n"
                   << "============================================================\n"
                   << "LIFTING BLOCK: 0x" << llvm::format_hex_no_prefix(addr, 16)
                   << "\n"
                   << "============================================================\n";

      if (!LiftFunctionAt(addr)) {
        return {};
      }
    }

    return entryFunctions;
  }

  void ConstifyProgramCounters() {

    auto *i64 = llvm::Type::getInt64Ty(module_.getContext());

    for (auto &kv : funcAt_) {
      kv.second->getArg(1)->replaceAllUsesWith(
          llvm::ConstantInt::get(i64, kv.first));
    }
  }

  std::vector<llvm::Function *> AllFunctions() const {

    std::vector<llvm::Function *> out;

    out.reserve(funcAt_.size());

    for (auto &kv : funcAt_) {
      out.push_back(kv.second);
    }

    return out;
  }

 private:

  const LiftRange *FindRange(uint64_t addr) const {

    for (const auto &range : ranges_) {
      if (addr >= range.start && (range.end == 0 || addr < range.end)) {
        return &range;
      }
    }

    return nullptr;
  }

  llvm::Function *GetOrDeclareFunction(uint64_t addr) {

    auto it = funcAt_.find(addr);

    if (it != funcAt_.end()) {
      return it->second;
    }

    char name[64];

    std::snprintf(name, sizeof(name), "block_%llx",
                  static_cast<unsigned long long>(addr));

    auto *func = arch_->DefineLiftedFunction(name, &module_);

    funcAt_[addr] = func;

    return func;
  }

  llvm::Function *GetOrScheduleFunction(uint64_t addr) {

    auto *func = GetOrDeclareFunction(addr);

    if (!visited_.count(addr)) {
      worklist_.push(addr);
    }

    return func;
  }

  void PrintInstructionInfo(
      uint64_t pc,
      const uint8_t *bytes,
      size_t available,
      const remill::Instruction &instruction) {

    size_t instructionSize = 0;

    if (instruction.next_pc > pc) {
      instructionSize = static_cast<size_t>(instruction.next_pc - pc);
    }

    if (instructionSize == 0 || instructionSize > available) {
      instructionSize = std::min<size_t>(available, 15);
    }

    llvm::outs() << "\n"
                 << "------------------------------------------------------------\n"
                 << "Address : 0x" << llvm::format_hex_no_prefix(pc, 16) << "\n"
                 << "Bytes   : ";

    for (size_t i = 0; i < instructionSize; ++i) {
      llvm::outs() << llvm::format_hex_no_prefix(
                          static_cast<unsigned>(bytes[i]), 2)
                   << " ";
    }

    llvm::outs() << "\n"
                 << "ASM     : ";

    capstone_.Print(pc, bytes, instructionSize);

    llvm::outs() << "Category: " << CategoryName(instruction.category) << "\n"
                 << "Next PC : 0x"
                 << llvm::format_hex_no_prefix(instruction.next_pc, 16) << "\n";

    using Category = remill::Instruction::Category;

    if (instruction.category == Category::kCategoryDirectJump ||
        instruction.category == Category::kCategoryDirectFunctionCall) {

      llvm::outs() << "Target  : 0x"
                   << llvm::format_hex_no_prefix(instruction.branch_taken_pc, 16)
                   << "\n";
    }

    if (instruction.category == Category::kCategoryConditionalBranch) {

      llvm::outs() << "Taken   : 0x"
                   << llvm::format_hex_no_prefix(instruction.branch_taken_pc, 16)
                   << "\n"
                   << "NotTaken: 0x"
                   << llvm::format_hex_no_prefix(
                          instruction.branch_not_taken_pc, 16)
                   << "\n";
    }

    llvm::outs() << "------------------------------------------------------------\n";
  }

  bool LiftFunctionAt(uint64_t addr) {

    llvm::Function *func = funcAt_[addr];

    llvm::BasicBlock *block = &func->getEntryBlock();

    const LiftRange *range = FindRange(addr);

    uint64_t rangeEnd = range ? range->end : 0;

    if (!range) {
      llvm::outs() << "[INFO] 0x" << llvm::format_hex_no_prefix(addr, 16)
                   << " is outside every specified LiftRange; "
                      "lifting it unbounded.\n";
    }

    uint64_t pc = addr;

    for (;;) {

      if (rangeEnd != 0 && pc >= rangeEnd) {

        llvm::outs() << "[RANGE END] 0x" << llvm::format_hex_no_prefix(pc, 16)
                     << "\n";

        llvm::IRBuilder<> ir(block);

        ir.CreateRet(remill::LoadMemoryPointer(block, intrinsics_));

        return true;
      }

      uint8_t bytes[16] = {0};

      size_t n = image_.Read(pc, bytes, sizeof(bytes));

      if (n == 0) {
        llvm::errs() << "No bytes available at 0x" << llvm::format_hex(pc, 10)
                     << "\n";
        return false;
      }

      std::string_view view(reinterpret_cast<const char *>(bytes), n);

      remill::Instruction instruction;

      remill::DecodingContext decoding_context = arch_->CreateInitialContext();

      if (!arch_->DecodeInstruction(pc, view, instruction, decoding_context)) {

        llvm::errs() << "Failed to decode instruction at 0x"
                     << llvm::format_hex(pc, 10) << "\n";

        llvm::outs() << "Capstone fallback: ";

        capstone_.Print(pc, bytes, n);

        return false;
      }

      instruction.is_atomic_read_modify_write = false;

      PrintInstructionInfo(pc, bytes, n, instruction);

      auto lifter = instruction.GetLifter();

      if (!lifter) {
        llvm::errs() << "No lifter for instruction at 0x"
                     << llvm::format_hex(pc, 10) << "\n";
        return false;
      }

      llvm::outs() << "Remill  : lifting... ";

      auto status = lifter->LiftIntoBlock(instruction, block);

      if (status != remill::kLiftedInstruction) {

        llvm::outs() << "FAILED\n";

        llvm::errs() << "Failed to lift instruction at 0x"
                     << llvm::format_hex(pc, 10) << "\n";

        return false;
      }

      llvm::outs() << "OK\n";

      using Category = remill::Instruction::Category;

      switch (instruction.category) {

        case Category::kCategoryNormal:
        case Category::kCategoryNoOp: {
          pc = instruction.next_pc;
          continue;
        }

        case Category::kCategoryDirectJump: {

          llvm::outs() << "\n[CONTROL FLOW]\nDirect jump -> 0x"
                       << llvm::format_hex_no_prefix(
                              instruction.branch_taken_pc, 16)
                       << "\n";

          auto *target = GetOrScheduleFunction(instruction.branch_taken_pc);

          remill::AddTerminatingTailCall(block, target, intrinsics_);

          return true;
        }

        case Category::kCategoryDirectFunctionCall: {

          llvm::outs() << "\n[CONTROL FLOW]\nDirect call -> 0x"
                       << llvm::format_hex_no_prefix(
                              instruction.branch_taken_pc, 16)
                       << "\nSimplified call handling: fall-through\n";

          pc = instruction.next_pc;

          continue;
        }

        case Category::kCategoryConditionalBranch: {

          llvm::outs() << "\n[CONTROL FLOW]\nConditional branch\n"
                       << "  TAKEN     -> 0x"
                       << llvm::format_hex_no_prefix(
                              instruction.branch_taken_pc, 16)
                       << "\n"
                       << "  NOT TAKEN -> 0x"
                       << llvm::format_hex_no_prefix(
                              instruction.branch_not_taken_pc, 16)
                       << "\n";

          auto *takenFunc = GetOrScheduleFunction(instruction.branch_taken_pc);

          auto *notTakenFunc =
              GetOrScheduleFunction(instruction.branch_not_taken_pc);

          auto *takenStub =
              llvm::BasicBlock::Create(func->getContext(), "taken", func);

          auto *notTakenStub =
              llvm::BasicBlock::Create(func->getContext(), "not_taken", func);

          auto found = remill::FindVarInFunction(func, "BRANCH_TAKEN", true);

          llvm::Value *cond = found.first;

          llvm::Type *condTy = found.second;

          if (cond) {

            llvm::outs() << "BRANCH_TAKEN variable: FOUND\n";

            llvm::IRBuilder<> ir(block);

            auto *loaded = ir.CreateLoad(condTy, cond);

            auto *pred = ir.CreateICmpNE(
                loaded, llvm::Constant::getNullValue(condTy));

            ir.CreateCondBr(pred, takenStub, notTakenStub);

          } else {

            llvm::errs() << "Warning: BRANCH_TAKEN not found at 0x"
                         << llvm::format_hex(pc, 10) << "\n"
                         << "Check Remill variable name for your version.\n"
                         << "Defaulting to fall-through edge.\n";

            llvm::IRBuilder<> ir(block);

            ir.CreateBr(notTakenStub);
          }

          remill::AddTerminatingTailCall(takenStub, takenFunc, intrinsics_);

          remill::AddTerminatingTailCall(
              notTakenStub, notTakenFunc, intrinsics_);

          return true;
        }

        case Category::kCategoryIndirectJump:
        case Category::kCategoryIndirectFunctionCall:
        case Category::kCategoryFunctionReturn: {

          llvm::outs() << "\n[CONTROL FLOW]\nDynamic control flow / return\n"
                          "Stopping static walk.\n";

          llvm::IRBuilder<> ir(block);

          ir.CreateRet(remill::LoadMemoryPointer(block, intrinsics_));

          return true;
        }

        default: {

          llvm::errs() << "Unhandled instruction category at 0x"
                       << llvm::format_hex(pc, 10) << ", treating as normal\n";

          pc = instruction.next_pc;

          continue;
        }
      }
    }
  }

 private:

  const remill::Arch *arch_;

  llvm::Module &module_;

  const remill::IntrinsicTable &intrinsics_;

  const PEImage &image_;

  std::vector<LiftRange> ranges_;

  CapstoneDisassembler capstone_;

  std::queue<uint64_t> worklist_;

  std::set<uint64_t> visited_;

  std::map<uint64_t, llvm::Function *> funcAt_;
};

static void StripModuleToFunction(llvm::Module &module, llvm::Function *keep) {

  for (const char *name : {"llvm.used", "llvm.compiler.used"}) {
    if (auto *g = module.getNamedGlobal(name)) {
      g->eraseFromParent();
    }
  }

  for (auto &f : module) {
    if (&f != keep && !f.isDeclaration()) {
      f.deleteBody();
      f.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }
  }

  bool erased = true;

  while (erased) {

    erased = false;

    for (auto it = module.global_begin(); it != module.global_end();) {
      llvm::GlobalVariable &g = *it;
      ++it;

      if (g.use_empty()) {
        g.eraseFromParent();
        erased = true;
      }
    }

    for (auto it = module.begin(); it != module.end();) {
      llvm::Function &f = *it;
      ++it;

      if (&f != keep && f.use_empty()) {
        f.eraseFromParent();
        erased = true;
      }
    }
  }
}

static bool EmitObjectFile(
    llvm::Module &module, llvm::Function *entry, const std::string &basePath) {

  StripModuleToFunction(module, entry);

  if (auto *ram = module.getGlobalVariable("RAM")) {
    ram->setInitializer(nullptr);
    ram->setLinkage(llvm::GlobalValue::ExternalLinkage);
  }

  const std::string modPath = basePath + "_module.ll";

  {
    std::error_code ec;

    llvm::raw_fd_ostream out(modPath, ec, llvm::sys::fs::OF_None);

    if (ec) {
      llvm::errs() << "Failed to open " << modPath << ": " << ec.message()
                   << "\n";
      return false;
    }

    module.print(out, nullptr);
  }

  if (llvm::verifyModule(module, &llvm::errs())) {
    llvm::errs() << "Stripped module verification failed!\n";
    return false;
  }

  std::string cmd = "llc -filetype=obj \"" + modPath + "\" -o \"" + basePath +
                    ".o\" 2> \"" + basePath + ".llc.log\"";

  int rc = std::system(cmd.c_str());

  if (rc != 0) {
    llvm::errs() << "llc failed or not found in PATH (see " << basePath
                 << ".llc.log)\n";
    return false;
  }

  llvm::outs() << "Object file written: " << basePath << ".o\n";

  return true;
}

static bool ParseAddress(const char *text, uint64_t &out) {

  if (!text || !*text) {
    return false;
  }

  errno = 0;

  char *end = nullptr;

  unsigned long long value = std::strtoull(text, &end, 16);

  if (errno != 0 || end == text || *end != '\0') {
    return false;
  }

  out = static_cast<uint64_t>(value);

  return true;
}

static void PrintUsage(const char *prog) {
  llvm::errs() << "Usage: " << prog
               << " <exe_path> <start_addr_hex> [end_addr_hex|0]"
                  " [hotpatch_bc_path]\n"
               << "Example: " << prog
               << " \"D:\\game\\game.exe\" 0x140001000 0x140002000\n";
}

int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  if (argc < 3) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string exePath = argv[1];

  LiftRange liftRange{0, 0};

  if (!ParseAddress(argv[2], liftRange.start)) {
    llvm::errs() << "Invalid start address: " << argv[2] << "\n";
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  if (argc > 3 && !ParseAddress(argv[3], liftRange.end)) {
    llvm::errs() << "Invalid end address: " << argv[3] << "\n";
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  if (liftRange.end != 0 && liftRange.end <= liftRange.start) {
    llvm::errs() << "End address must be greater than start address\n";
    return EXIT_FAILURE;
  }

  if (!std::filesystem::exists(exePath)) {
    llvm::errs() << "File not found: " << exePath << "\n";
    return EXIT_FAILURE;
  }

  const std::vector<LiftRange> functionRanges = {liftRange};

  llvm::outs() << "Loading PE: " << exePath << "\n";

  PEImage image(exePath);

  llvm::outs() << "ImageBase: 0x"
               << llvm::format_hex_no_prefix(image.ImageBase(), 16) << "\n"
               << "EntryPoint: 0x"
               << llvm::format_hex_no_prefix(image.EntryPoint(), 16) << "\n";

  llvm::LLVMContext context;

  auto arch = remill::Arch::Get(context, "windows", "amd64");

  if (!arch) {
    llvm::outs() << "Failed to get architecture\n";
    return EXIT_FAILURE;
  }

  auto semantics = remill::LoadArchSemantics(arch.get());

  if (!semantics) {
    llvm::outs() << "Failed to load architecture semantics\n";
    return EXIT_FAILURE;
  }

  auto hotpatchPath =
      argc > 4 ? std::filesystem::path(argv[4])
               : executableDir() / "helpers\\x86_64\\RemillHelpers.bc";

  if (std::filesystem::exists(hotpatchPath)) {

    llvm::outs() << "Applying hotpatch from: " << hotpatchPath.string() << "\n";

    if (!hotpatchRemill(*semantics, hotpatchPath.string())) {
      llvm::outs() << "Warning: Failed to apply hotpatch\n";
    }

  } else {
    llvm::outs() << "No hotpatch file found at: " << hotpatchPath.string()
                 << "\n";
  }

  auto intrinsics = arch->GetInstrinsicTable();

  if (!intrinsics) {
    llvm::outs() << "Failed to get intrinsic table\n";
    return EXIT_FAILURE;
  }

  llvm::outs() << "\n"
               << "##################################################################\n"
               << "# START LIFTING\n"
               << "##################################################################\n";

  for (const auto &range : functionRanges) {
    llvm::outs() << "Range: 0x" << llvm::format_hex_no_prefix(range.start, 16)
                 << " .. 0x" << llvm::format_hex_no_prefix(range.end, 16)
                 << "\n";
  }

  RangeLifter rangeLifter(
      arch.get(), *semantics, *intrinsics, image, functionRanges);

  auto entryFunctions = rangeLifter.Run();

  if (entryFunctions.empty()) {
    llvm::errs() << "Lifting failed\n";
    return EXIT_FAILURE;
  }

  auto allFunctions = rangeLifter.AllFunctions();

  rangeLifter.ConstifyProgramCounters();

  llvm::outs() << "\n"
               << "##################################################################\n"
               << "# LIFTING COMPLETE\n"
               << "##################################################################\n"
               << "Lifted " << allFunctions.size() << " block(s) from "
               << entryFunctions.size() << " requested range(s)\n";

  llvm::Function *entryFunction = nullptr;

  llvm::outs() << "\nRewriting RAM accesses...\n";

  ramclean::Rewrite(*semantics,
                    kKeepRamGlobal ? ramclean::Mode::KeepGlobal
                                   : ramclean::Mode::IntToPtr);

  llvm::outs() << "Running Remill optimizer...\n";

  remill::OptimizeModule(arch.get(), semantics.get(), allFunctions);

  entryFunction = BuildMultiEntryWrapper(*semantics, entryFunctions);

  if (entryFunctions.size() > 1) {
    allFunctions.push_back(entryFunction);
  }

  if (kSSAState) {
    entryFunction = BuildSSAWrapper(arch.get(), *semantics, entryFunction);
    allFunctions.push_back(entryFunction);
  }

  entryFunction =
      ForceInlineLiftedBlocks(*semantics, entryFunction, allFunctions);

  entryFunction->addParamAttr(0, llvm::Attribute::NoAlias);

  AnnotateFlagRanges(arch.get(), *entryFunction);

  llvm::outs() << "Running post-inline simplification...\n";

  aggressiveSimplify(entryFunction);

  if (kAssumeNoRedZone && kKeepRamGlobal) {
    if (RemoveDeadStackStores(arch.get(), *entryFunction)) {
      aggressiveSimplify(entryFunction);
    }
  }

  if (kLowerRamAtEnd && kKeepRamGlobal) {
    ramclean::Rewrite(*semantics, ramclean::Mode::IntToPtr);
    aggressiveSimplify(entryFunction);
  }

  llvm::outs() << "\n"
               << "##################################################################\n"
               << "# LIFTED FUNCTION IR\n"
               << "##################################################################\n";

  entryFunction->print(llvm::outs());

  if (!SaveLiftedFunctionIR(*entryFunction, "lifted_output")) {
    return EXIT_FAILURE;
  }

  if (!EmitObjectFile(*semantics, entryFunction, "lifted_output")) {
    llvm::errs() << "Warning: .o emission failed\n";
  }

  llvm::outs() << "\n"
               << "##################################################################\n"
               << "# DONE\n"
               << "##################################################################\n";

  return EXIT_SUCCESS;
}
