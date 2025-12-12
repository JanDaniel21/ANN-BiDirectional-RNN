//======================================================
// Name: <Aser S. Baladjay>
// Date: 2025-12-12
// Program: Fibonacci Sequence Generator
// Description:
//   Reads N from RAM[0] and writes the first N Fibonacci
//   numbers into consecutive memory starting at RAM[100].
//   If N <= 0, the program does nothing.
//   If N > 20, N is clamped to 20 to avoid overflow.
//======================================================

//------------------------------------------------------
// Symbol table (variables)
//   N    : requested count of Fibonacci numbers
//   I    : loop counter (how many numbers written)
//   F0   : previous Fibonacci number
//   F1   : current Fibonacci number
//   ADDR : current RAM address to write to (starts at 100)
//   TMP  : temporary register for sums
//------------------------------------------------------

// Load N from RAM[0] into variable N
@0
D=M        // D = RAM[0] (user-provided N)
@N
M=D        // N = RAM[0]

// If N <= 0, jump to END (nothing to compute)
@N
D=M        // D = N
@END
D;JLE      // if N <= 0, go to END

// Clamp N to 20 if N > 20
@N
D=M        // D = N
@20
D=D-A      // D = N - 20
@NOCLAMP
D;JLE      // if N - 20 <= 0, i.e., N <= 20, skip clamp

// Here N > 20, so set N = 20
@20
D=A
@N
M=D

(NOCLAMP)
// Initialize I = 0
@I
M=0

// Initialize first two Fibonacci values: F0 = 0, F1 = 1
@F0
M=0
@F1
M=1

// Initialize ADDR = 100 (output start address)
@100
D=A
@ADDR
M=D

//------------------------------------------------------
// Write first Fibonacci number (F0) if N >= 1
//------------------------------------------------------
@F0
D=M        // D = F0 (0)
@ADDR
A=M        // A = current output address
M=D        // RAM[ADDR] = F0

// ADDR++
@ADDR
M=M+1

// I++
@I
M=M+1

// If I >= N, we are done
@I
D=M
@N
D=D-M      // D = I - N
@END
D;JGE      // if I >= N, jump to END

//------------------------------------------------------
// Write second Fibonacci number (F1) if N >= 2
//------------------------------------------------------
@F1
D=M        // D = F1 (1)
@ADDR
A=M
M=D        // RAM[ADDR] = F1

// ADDR++
@ADDR
M=M+1

// I++
@I
M=M+1

// If I >= N, we are done
@I
D=M
@N
D=D-M      // D = I - N
@END
D;JGE      // if I >= N, jump to END

//------------------------------------------------------
// Main loop: compute and write remaining Fibonacci numbers
//------------------------------------------------------
(LOOP)
// TMP = F0 + F1
@F0
D=M        // D = F0
@F1
D=D+M      // D = F0 + F1
@TMP
M=D        // TMP = F0 + F1

// F0 = F1
@F1
D=M
@F0
M=D

// F1 = TMP
@TMP
D=M
@F1
M=D

// Write F1 to RAM[ADDR]
@F1
D=M
@ADDR
A=M
M=D

// ADDR++
@ADDR
M=M+1

// I++
@I
M=M+1

// If I < N, continue loop
@I
D=M
@N
D=D-M      // D = I - N
@LOOP
D;JLT      // if I < N, jump back to LOOP

//------------------------------------------------------
// END: Halt by jumping to itself
//------------------------------------------------------
(END)
@END
0;JMP

//======================================================
// Variable declarations (resolved by assembler to RAM
// addresses starting at 16).
//======================================================
(N)
(I)
(F0)
(F1)
(ADDR)
(TMP)
