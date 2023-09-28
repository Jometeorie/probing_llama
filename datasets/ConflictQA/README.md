---
license: apache-2.0
task_categories:
- question-answering
language:
- en
pretty_name: conflictQA
size_categories:
- 10K<n<100K
---

# Dataset Card for ConflcitQA
## Dataset Description
- **Repository:** https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict
- **Paper:** https://arxiv.org/abs/2305.13300
- **Point of Contact:** Point of Contact: [Jian Xie](mailto:jianx0321@gmail.com)
## Citation
If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.
```bib
@article{xie2023adaptive,
  title={Adaptive Chameleon or Stubborn Sloth: Unraveling the Behavior of Large Language Models in Knowledge Conflicts},
  author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Lou, Renze and Su, Yu},
  journal={arXiv preprint arXiv:2305.13300},
  url={arxiv.org/abs/2305.13300},
  year={2023}
}
```

# ConflcitQA

We provide the conflictQA GPT-4 (ChatGPT) version, which utilizes GPT-4 (ChatGPT) guided parametric memory.

```json
{"question": "What is George Rankin's occupation?", "popularity": 142, "ground_truth": ["politician", "political leader", "political figure", "polit.", "pol"], "memory_answer": "George Rankin's occupation is a professional photographer.", "parametric_memory": "As a professional photographer, George Rankin...", "counter_answer": "George Rankin's occupation is political figure.", "counter_memory": "George Rankin has been actively involved in politics for over a decade...", "parametric_memory_aligned_evidence": "George Rankin has a website showcasing his photography portfolio...", "counter_memory_aligned_evidence": "George Rankin Major General George James Rankin..."}
```

# Data Fields
- "question": The question in natural language
- "popularity": The monthly page views on Wikipedia for the given question
- "ground_truth": The factual answer to the question, which may include multiple possible answers
- "memory_answer": The answer provided by the LLM to the question
- "parametric_memory": The supportive evidence from LLM's parametric memory for the answer
- "counter_answer": The answer contradicting the "memory_answer"
- "counter_memory": The generation-based evidence supporting the counter_answer
- "parametric_memory_aligned_evidence": Additional evidence supporting the "memory_answer", which could be generated or derived from Wikipedia/human annotation
- "counter_memory_aligned_evidence": Additional evidence supporting the "counter_answer", either generated or sourced from Wikipedia/human annotation


