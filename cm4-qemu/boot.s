.syntax unified
.cpu cortex-m4
.thumb

.section .isr_vector
    .long   __StackTop
    .long   Reset_Handler
    .long   Default_Handler

.text
.thumb_func
.global Reset_Handler
Reset_Handler:
    @ ldr   R0, = main
    bl SystemInit
    ldr R0, =_start
    bx R0

.thumb_func
.global Default_Handler
Default_Handler:
hang:
    b hang