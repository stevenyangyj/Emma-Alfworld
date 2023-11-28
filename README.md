# EMMA

This repository is the official implementation of the following paper.

**[Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld]()**
<br/>
[Yijun Yang](https://scholar.google.com/citations?user=X0quXnsAAAAJ&hl=en), [Tainyi Zhou](https://tianyizhou.github.io/), Kanxue Li, [Dapeng Tao](https://scholar.google.com/citations?user=AQzS40gAAAAJ&hl=en), Lvsong Li, [Li Shen](https://sites.google.com/site/mathshenli/home), [Xiaodong He](https://scholar.google.com/citations?user=W5WbqgoAAAAJ&hl=en), [Jing Jiang](https://profiles.uts.edu.au/Jing.Jiang), [Yuhui Shi](https://scholar.google.com/citations?user=xSvAHWgAAAAJ&hl=en)
<br/>

[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/mit/) [![arXiv](https://img.shields.io/badge/arXiv-2311.14603-b31b1b.svg)]()


![](assets/agent_architecture.png)


## Abstract
> While large language models (LLMs) excel in a simulated world of texts, they struggle to interact with the more realistic world without perceptions of other modalities such as visual or audio signals. Although vision-language models (VLMs) integrate LLM modules (1) aligned with static image features, and (2) may possess prior knowledge of world dynamics (as demonstrated in the text world), they have not been trained in an embodied visual world and thus cannot align with its dynamics. On the other hand, training an embodied agent in a noisy visual world without expert guidance is often challenging and inefficient. In this paper, we train a VLM agent living in a visual world using an LLM agent excelling in a parallel text world (but inapplicable to the visual world). Specifically, we distill LLM's reflection outcomes (improved actions by analyzing mistakes) in a text world's tasks to finetune the VLM on the same tasks of the visual world, resulting in an Embodied Multi-Modal Agent (EMMA) quickly adapting to the visual world dynamics. Such cross-modality imitation learning between the two parallel worlds enables EMMA to generalize to a broad scope of new tasks without any further guidance from the LLM expert. Extensive evaluations on the ALFWorld benchmark highlight EMMA's superior performance to SOTA VLM-based agents across diverse tasks, e.g., 20%-70% improvement in the success rate.


## TODO
- [ ] Release code very soon
