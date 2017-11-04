/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * This software may be used and distributed according to the terms of the GNU
 * General Public License version 2 or any later version.
 */

#include <cstring>
#include <limits>

#include <xbyak.h>

#include "backend_x64/abi.h"
#include "backend_x64/block_of_code.h"
#include "backend_x64/jitstate.h"
#include "common/assert.h"
#include "dynarmic/callbacks.h"

namespace Dynarmic {
namespace BackendX64 {

#ifdef _WIN32
const Xbyak::Reg64 BlockOfCode::ABI_RETURN = Xbyak::util::rax;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM1 = Xbyak::util::rcx;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM2 = Xbyak::util::rdx;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM3 = Xbyak::util::r8;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM4 = Xbyak::util::r9;
#else
const Xbyak::Reg64 BlockOfCode::ABI_RETURN = Xbyak::util::rax;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM1 = Xbyak::util::rdi;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM2 = Xbyak::util::rsi;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM3 = Xbyak::util::rdx;
const Xbyak::Reg64 BlockOfCode::ABI_PARAM4 = Xbyak::util::rcx;
#endif

constexpr size_t TOTAL_CODE_SIZE = 128 * 1024 * 1024;
constexpr size_t FAR_CODE_OFFSET = 100 * 1024 * 1024;

BlockOfCode::BlockOfCode(UserCallbacks cb, LookupBlockCallback lookup_block, void* lookup_block_arg)
        : Xbyak::CodeGenerator(TOTAL_CODE_SIZE)
        , cb(cb)
        , lookup_block(lookup_block)
        , lookup_block_arg(lookup_block_arg)
        , constant_pool(this, 256)
{
    GenRunCode();
    GenMemoryAccessors();
    exception_handler.Register(this);
    near_code_begin = getCurr();
    far_code_begin = getCurr() + FAR_CODE_OFFSET;
    ClearCache();
}

void BlockOfCode::ClearCache() {
    in_far_code = false;
    near_code_ptr = near_code_begin;
    far_code_ptr = far_code_begin;
    SetCodePtr(near_code_begin);
}

size_t BlockOfCode::RunCode(JitState* jit_state, size_t cycles_to_run) const {
    constexpr size_t max_cycles_to_run = static_cast<size_t>(std::numeric_limits<decltype(jit_state->cycles_remaining)>::max());
    ASSERT(cycles_to_run <= max_cycles_to_run);

    jit_state->cycles_remaining = cycles_to_run;
    run_code(jit_state);
    return cycles_to_run - jit_state->cycles_remaining; // Return number of cycles actually run.
}

void BlockOfCode::ReturnFromRunCode(bool MXCSR_switch) {
    size_t index = 0;
    if (!MXCSR_switch)
        index |= NO_SWITCH_MXCSR;
    jmp(return_from_run_code[index]);
}

void BlockOfCode::ForceReturnFromRunCode(bool MXCSR_switch) {
    size_t index = FORCE_RETURN;
    if (!MXCSR_switch)
        index |= NO_SWITCH_MXCSR;
    jmp(return_from_run_code[index]);
}

void BlockOfCode::GenRunCode() {
    Xbyak::Label loop;

    align();
    run_code = getCurr<RunCodeFuncType>();

    // This serves two purposes:
    // 1. It saves all the registers we as a callee need to save.
    // 2. It aligns the stack so that the code the JIT emits can assume
    //    that the stack is appropriately aligned for CALLs.
    ABI_PushCalleeSaveRegistersAndAdjustStack(this);

    mov(r15, ABI_PARAM1);

    L(loop);
    mov(ABI_PARAM1, u64(lookup_block_arg));
    CallFunction(lookup_block);

    SwitchMxcsrOnEntry();
    jmp(ABI_RETURN);

    // Return from run code variants
    const auto emit_return_from_run_code = [this, &loop](bool no_mxcsr_switch, bool force_return){
        if (!no_mxcsr_switch) {
            SwitchMxcsrOnExit();
        }

        if (!force_return) {
            cmp(qword[r15 + offsetof(JitState, cycles_remaining)], 0);
            jg(loop);
        }

        ABI_PopCalleeSaveRegistersAndAdjustStack(this);

        // As we do not know if user-code is AVX or SSE, an AVX-SSE transition may occur.
        MaybeAvxToSseTransition();

        ret();
    };

    align();
    return_from_run_code[0] = getCurr<const void*>();
    emit_return_from_run_code(false, false);

    align();
    return_from_run_code[NO_SWITCH_MXCSR] = getCurr<const void*>();
    emit_return_from_run_code(true, false);

    align();
    return_from_run_code[FORCE_RETURN] = getCurr<const void*>();
    emit_return_from_run_code(false, true);

    align();
    return_from_run_code[NO_SWITCH_MXCSR | FORCE_RETURN] = getCurr<const void*>();
    emit_return_from_run_code(true, true);
}

void BlockOfCode::GenMemoryAccessors() {
    align();
    read_memory_8 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Read8);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();

    align();
    read_memory_16 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Read16);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();

    align();
    read_memory_32 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Read32);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();

    align();
    read_memory_64 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Read64);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();

    align();
    write_memory_8 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Write8);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();

    align();
    write_memory_16 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Write16);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();

    align();
    write_memory_32 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Write32);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();

    align();
    write_memory_64 = getCurr<const void*>();
    ABI_PushCallerSaveRegistersAndAdjustStack(this);
    CallFunction(cb.memory.Write64);
    ABI_PopCallerSaveRegistersAndAdjustStack(this);
    ret();
}

void BlockOfCode::SwitchMxcsrOnEntry() {
    AVX(this, stmxcsr, dword[r15 + offsetof(JitState, save_host_MXCSR)]);
    AVX(this, ldmxcsr, dword[r15 + offsetof(JitState, guest_MXCSR)]);
}

void BlockOfCode::SwitchMxcsrOnExit() {
    AVX(this, stmxcsr, dword[r15 + offsetof(JitState, guest_MXCSR)]);
    AVX(this, ldmxcsr, dword[r15 + offsetof(JitState, save_host_MXCSR)]);
}

void BlockOfCode::MaybeAvxToSseTransition() {
    // There are two kinds of transition penalties.
    // The first is when transitioning from AVX to legacy SSE code.
    // The second is when transitioning from legacy SSE to AXV code.
    // The second penalty only occurs if the first had occured previously.
    //
    // This occurs because the YMM registers can be in three states:
    // A. Upper 128-bits are known to be zero.
    // B. Full 256-bits are used. ("Dirty upper" state.)
    // C. Upper 128-bits are saved, lower 128-bits currently used. ("Preserved upper".)
    //
    // AVX instructions require the CPU to be in state B.
    // SSE instructions require the CPU to be in state A or C.
    // Transitions between A and B are cheap (and are done with vzeroupper/vzeroall).
    // Transitions to and from C are expensive.
    //
    // Depending on microarchitecture this may be tracked at register granularity.
    //
    // Sandy Bridge to Broadwell: One-time penalty of about 70 cycles.
    // Skylake: State C doesn't exist as SSE instructions can execute in state B,
    //          but doing so has an ongoing penalty.
    // Knights Landing: Intel recommends not emitting vzeroupper on this microarch.
    //
    // AMD up to Ryzen: No known penalty for mixing AVX and non-AVX instructions.

    if (DoesCpuSupport(Xbyak::util::Cpu::tAVX)) {
        vzeroupper();
    }
}

Xbyak::Address BlockOfCode::MConst(u64 constant) {
    return constant_pool.GetConstant(constant);
}

void BlockOfCode::SwitchToFarCode() {
    ASSERT(!in_far_code);
    in_far_code = true;
    near_code_ptr = getCurr();
    SetCodePtr(far_code_ptr);

    ASSERT_MSG(near_code_ptr < far_code_begin, "Near code has overwritten far code!");
}

void BlockOfCode::SwitchToNearCode() {
    ASSERT(in_far_code);
    in_far_code = false;
    far_code_ptr = getCurr();
    SetCodePtr(near_code_ptr);
}

void* BlockOfCode::AllocateFromCodeSpace(size_t alloc_size) {
    if (size_ + alloc_size >= maxSize_) {
        throw Xbyak::Error(Xbyak::ERR_CODE_IS_TOO_BIG);
    }

    void* ret = getCurr<void*>();
    size_ += alloc_size;
    memset(ret, 0, alloc_size);
    return ret;
}

void BlockOfCode::SetCodePtr(CodePtr code_ptr) {
    // The "size" defines where top_, the insertion point, is.
    size_t required_size = reinterpret_cast<const u8*>(code_ptr) - getCode();
    setSize(required_size);
}

void BlockOfCode::EnsurePatchLocationSize(CodePtr begin, size_t size) {
    size_t current_size = getCurr<const u8*>() - reinterpret_cast<const u8*>(begin);
    ASSERT(current_size <= size);
    nop(size - current_size);
}

bool BlockOfCode::DoesCpuSupport(Xbyak::util::Cpu::Type type) const {
    return cpu_info.has(type);
}

bool BlockOfCode::ShouldEmitAvx() const {
    return DoesCpuSupport(Xbyak::util::Cpu::tAVX);
}

} // namespace BackendX64
} // namespace Dynarmic
