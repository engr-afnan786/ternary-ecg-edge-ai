.syntax unified
.cpu cortex-m4
.thumb

.section .isr_vector,"a",%progbits
.balign 128
.word __stack_top
.word Reset_Handler
.rept 14
.word Default_Handler
.endr

.section .text.Reset_Handler,"ax",%progbits
.thumb_func
.global Reset_Handler
Reset_Handler:
    cpsid i
    ldr r0, =__data_load
    ldr r1, =__data_start
    ldr r2, =__data_end
1:  cmp r1, r2
    bhs 2f
    ldr r3, [r0], #4
    str r3, [r1], #4
    b 1b
2:  ldr r1, =__bss_start
    ldr r2, =__bss_end
    movs r3, #0
3:  cmp r1, r2
    bhs 4f
    str r3, [r1], #4
    b 3b
4:  bl firmware_main
.thumb_func
Default_Handler:
    b Default_Handler

/* Arm semihosting uses operation in r0, argument address in r1. */
.thumb_func
.global semihost
semihost:
    bkpt 0xab
    bx lr
