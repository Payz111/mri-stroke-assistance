# Model Card: MRI Stroke Segmentation v1.0

## Model Details

- **Architecture:** 3D U-Net
- **Training data:** ISLES 2022 (250 cases, multi-center)
- **Input:** DWI + ADC + FLAIR (3-channel, 3D volume)
- **Output:** Lesion probability map

## Intended Use

- Research and development
- Clinical decision support (with physician oversight)
- Educational purposes

## Limitations

- Trained on limited dataset (250 cases)
- May underperform on:
  - Very small lesions (< 1 ml)
  - Posterior fossa lesions
  - Non-standard MRI protocols
- NOT validated for clinical use
- NOT a medical device

## Ethical Considerations

- Requires human expert review for all outputs
- Should not be used for autonomous diagnosis
- No patient data retained in model weights

## Metrics

*To be filled after training.*

## Training Details

*To be filled after training.*
