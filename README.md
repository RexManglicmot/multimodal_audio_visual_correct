## Question
Can we use multimodal data—(1) egocentric **video** from a body-worn camera and (2) synchronized ambient **audio**—to accurately detect whether a short clip corresponds to a fall vs non-fall event, using only a subset of 2,000 clips from the EGOFALLS dataset?


## Dataset



Transformations...labeling...annotations




## Metrics
This project defines each **clip** as a short, synchronized audio–video segment from the EGOFALLS dataset, recorded from an egocentric (body-worn) camera with ambient audio and labeled as either *fall* or *non-fall*. All metrics below are computed over these clips, treating **fall** as the positive class.

* **Loss** – Training objective the model minimizes; lower loss means predictions match the ground-truth labels better.
* **Accuracy** – Proportion of all clips (falls + non-falls) that are classified correctly.
* **Precision (Fall)** – Of all clips predicted as *falls*, the fraction that are actually falls (how many alarms are correct).
* **Recall (Fall)** – Of all true *fall* clips, the fraction the model correctly detects (how many falls we catch).
* **F1 (Fall)** – Harmonic mean of precision and recall for the fall class; high only when both are strong.
* **ROC AUC (Fall)** – Area under the ROC curve treating “fall” as the positive class; measures how well the model separates falls from non-falls across thresholds.
* **PR AUC (Fall)** – Area under the precision–recall curve for the fall class; especially informative when falls are relatively rare.
* **TP / FP / FN / TN** – Confusion matrix counts: true positives (correct falls), false positives (non-falls flagged as falls), false negatives (missed falls), and true negatives (correct non-falls).



## Results
| Mode       | Loss  | Accuracy | F1 (Fall) | Recall (Fall) | Precision (Fall) | ROC AUC (Fall) | TP | FP | FN | TN |
| ---------- | ----- | -------- | --------- | ------------- | ---------------- | -------------- | -- | -- | -- | -- |
| fusion     | 0.179 | **0.944**    | **0.945**     | **1.000**         | **0.897**            | 0.977          | 26 | 3  | 0  | 25 |
| video_only | 0.426 | 0.870    | 0.877     | 0.962         | 0.806            | 0.887          | 25 | 6  | 1  | 22 |
| audio_only | 0.505 | 0.722    | 0.776     | 1.000         | 0.634            | 0.713          | 26 | 15 | 0  | 13 |

The fusion model clearly performs best, achieving the highest accuracy (0.944) and perfect recall for falls (1.000) with strong precision (0.897), meaning it catches all falls while keeping false alarms relatively low. The video-only model is decent but noticeably worse, especially in precision and overall accuracy, while the audio-only model lags behind both, with much lower accuracy (0.722) and a high number of false positives, making it less reliable on its own.


![F1 for fall detection by mode](outputs/figures/bar_f1_fall_by_mode.png)
The Fall vs Non-fall F1 plot shows that multimodal fusion (audio + video) gives the strongest fall detection performance, with an F1 of 0.95, clearly above either single modality. Video-only does reasonably well at 0.88, but still underperforms fusion, while audio-only trails at 0.78, suggesting that audio alone misses more of the nuanced patterns needed for reliable fall detection. Overall, combining modalities consistently tightens performance and reduces the gap between precision and recall for the fall class.

![ROC curves for fall vs non-fall](outputs/figures/roc_curves_fall_all_modes.png)
The Fall vs Non-fall ROC curves show that the fusion model delivers the most robust discrimination between falls and non-falls, with an AUC of 0.98, staying close to the top-left corner across thresholds. The video-only model performs reasonably well (AUC ≈ 0.89), while the audio-only model lags behind (AUC ≈ 0.71), indicating substantially more overlap between fall and non-fall predictions when relying on audio alone.

![Precision–Recall curves for fall class](outputs/figures/pr_curves_fall_all_modes.png)
The Fall vs Non-fall PR curves highlight that the fusion model not only detects falls but does so with consistently high precision, achieving a PR AUC of 0.97 and maintaining strong precision even at high recall. The video-only model is noticeably weaker (PR AUC ≈ 0.76), and the audio-only model struggles the most (PR AUC ≈ 0.57), confirming that multimodal fusion is especially valuable when we care about catching falls without flooding the system with false alarms.

## Conclusion
The experiments show that **combining egocentric video with ambient audio clearly outperforms single-modality models for fall detection**. The fusion model achieves the highest F1, ROC AUC, and PR AUC, meaning it can **catch nearly all falls while keeping false alarms relatively low**, which is critical for any real-time alerting system. Video alone remains reasonably strong but consistently trails fusion, while audio alone is not reliable enough to be used in isolation. Overall, these results support using **multimodal sensing and late fusion** as a practical design choice for robust, wearable fall-detection systems, with future work focusing on scaling to larger datasets and more diverse environments.


## Next Steps
- Integrate richer audio–visual features, such as motion trajectories, optical flow, or learned audio embeddings (e.g., log-mel CNN features), to see if more informative representations can further boost fusion performance without drastically increasing complexity.

- Build a small evaluation + visualization dashboard (e.g., in Streamlit) that lets you browse clips, predicted labels, confidence scores, and confusion cases, making it easier to do qualitative error analysis and communicate results to non-technical stakeholders.

### Build Order
config.yaml ->