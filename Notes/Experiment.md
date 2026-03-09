**Experiment**

1. Success Rate evaluation across different tasks and different widths
   1. can also categorized based on task horizon (see SimpleVLA-RL paper https://arxiv.org/pdf/2509.09674)
   2. this means we can further have a discussion about if VLA planning burden is more aligned with width or aligned with plan length (given that both width and plan length is a metric for planning complexity of tasks)
      1. but wait, but this is a bit problematic, because I used oracle instructor to provide intermediate instructions, so some reviewer may argue that your instruction already decompose the task and VLA handles plan much shorter if instruction presented. in this way, how to justify? or we use the best setting for `test_single_instr_lang_ood` eval result to imply? 
2. With instruction vs Without instruction evaluation (evaluate over-reliance on visual observations)
3. attention pattern evaluation, how action token attend to previous observation tokens or language tokens
   1. ISSUE: different layers, different head

![image-20260107135549893](/home/sukaih/.config/Typora/typora-user-images/image-20260107135549893.png)

4. number of rules activation from  coming from the instructor
   1. if task failed and number of activation is larger than reference number -> it means agent undo achieved subgoals 
   2. if task failed and number of activation is smaller -> it means agent get stuck

5. Qualitative analysis on failure cases 