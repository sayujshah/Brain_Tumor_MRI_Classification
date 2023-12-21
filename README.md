# Brain Tumor MRI Classification Model

While a person has a less than 1% chance at developing a brain tumor in their lifetime, they account for 85% - 90% of
all primary central nervous system (CNS) tumors[^1]. There are numerous ways to diagnose a brain tumor. One of which
is a biopsy. Open surgery biopsies are highly accurate at nearly 100%, but very expensive and risky for the patient. On
the other hand, a safer approach is to conduct a fine-needle aspiration, in which a small amount of soft tissue is extracted
from the patient and a radiologist runs tests on said tissue to confirm whether there is a tumor or not. However, this too
can lead to excessive bleeding for the patient and has a very low 49.1% accuracy[^2]. Thanks to medical and technological
advancements, most radiologists now opt to utilize magnetic reasoning imaging (MRIs) to capture a high-quality scan of
a patient’s brain.

Many doctors still continue to diagnose brain tumors manually via an MRI scan. This is prone to misdiagnosis due to
the high complexities of brain tumors, which can lead to costly, time-consuming additional steps as well as life-altering
consequences. There are many different types of brain tumors that all have varying implications for a patient. Correctly
classifying *which* specific type of brain tumor a patient has is very important as it can provide valuable information on
what kind of procedure (if any) is needed, what type of cuts to make during a cancer operation, what specific sides of the
brain need the most attention, etc. Due to the large proportion of CNS tumors consisting of brain tumors and the increased
data captured through MRIs, we have a plethora of data available to potentially train machine learning models to classify
tumors ahead of time and release some burden from a radiologist making a final call. This can be a useful additional tool
employed by hospitals in addition to their current procedures in diagnosing such events.

In this repository, I have developed a brain tumor classification model with a **94%** accuracy at classifying the three most common brain tumor types: glioma, meningioma, and pituitary. It will also let you know if there is *no* tumor identified in the MRI. See below for a breakdown of the files you will find in this repository.

## Data

This folder contains 3,264 MRI scans split into the following classifications: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. The data is necessary to run the files used to train my models found in the **Model Training** folder. Feel free to download this and train your own models!

## Model Training

I trained various models ranging from a custom CNN model, decision tree, random forest, and SVM. However, none of these performed as well as the EfficientNetB0 and Microsofts's ResNet50 models pre-trained on the ImageNet database. In order for the model to train properly on the MRI data, I added more output layers to the base model (EfficienNetB0 or ResNet50).

### ResNet50

The ResNet50 model achieved a solid validation accuracy of **90.14%**. See below for the classification report and accuracy-per-epoch chart:

**Validation Accuracy:** 90.14%\
**MSE:** 0.1682\
**R^2:** 0.8833

|              | **precision** | **recall** | **f1-score** | **support** |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.87      | 0.86   | 0.87     | 103     |
| 1            | 0.84      | 0.82   | 0.83     | 90      |
| 2            | 0.93      | 0.93   | 0.93     | 40      |
| 3            | 0.96      | 0.99   | 0.97     | 94      |
| **accuracy**     |           |        | 0.90     | 327     |
| **macro avg**    | 0.90      | 0.90   | 0.90     | 327     |
| **weighted avg** | 0.90      | 0.90   | 0.90     | 327     |

![resnet](https://github.com/sayujshah/Brain_Tumor_MRI_Classification/assets/64810038/e584f46b-e6d6-45ce-8d17-f042a488c369)

### EfficientNetB0

The EfficientNetB0 performed much better with a validation accuracy of **94.22%**! See below for the classification report and accuracy-per-epoch chart:

**Validation Accuracy:** 94.22%\
**MSE:** 0.1284\
**R^2:** 0.9102

|              | **precision** | **recall** | **f1-score** | **support** |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.93      | 0.89   | 0.91     | 99      |
| 1            | 0.88      | 0.86   | 0.87     | 85      |
| 2            | 0.92      | 0.96   | 0.94     | 48      |
| 3            | 0.96      | 1.00   | 0.98     | 95      |
| **accuracy**     |           |        | 0.92     | 327     |
| **macro avg**    | 0.92      | 0.93   | 0.92     | 327     |
| **weighted avg** | 0.92      | 0.92   | 0.92     | 327     |

![effnet](https://github.com/sayujshah/Brain_Tumor_MRI_Classification/assets/64810038/a3a870b4-522a-4bcf-af14-426d5b7a2169)

Both models were not particularly great when classifying meningioma tumors. However, the EfficientNetB0 model was much better at avoiding false positives. You can train these models yourself and compare results by running the files found in the **Model Training** folder. Feel free to make some tweaks to improve performance!

## Brain_Tumor_MRI_Classification.py

This is the main file. If you just want to see the classification model in action, run this file. Once run, a webpage will open prompting you to upload your own MRI scan.

![mri_demo1](https://github.com/sayujshah/Brain_Tumor_MRI_Classification/assets/64810038/9d55c9d2-5e13-4167-8526-d9692425bc13)

For this demo, I found a novel image of an MRI scan of a pituitary tumor that the model was *not* trained on. Go ahead and upload any MRI scan you have an observe the results.

![mri_demo2](https://github.com/sayujshah/Brain_Tumor_MRI_Classification/assets/64810038/6626c455-d4cc-43c6-9f85-c04ad2d46947)

Once you upload the scan, click "Submit" and watch the model provide its best guess as to what type of tumor (if any) is present in the scan. We see that the model correctly predicted that my scan contained a pituitary tumor!

![mri_demo3](https://github.com/sayujshah/Brain_Tumor_MRI_Classification/assets/64810038/878b0bce-4102-4039-b094-8392ffb47a6d)

I chose to use the EfficientNetB0 model as it performed the best during my training, however you can edit the file to import your own custom model. Experiment with your own models and provide feedback with the buttons below the output.

[^1]: “Brain Tumor: Statistics.” *Cancer.Net*, 31 May 2023, www.cancer.net/cancer-types/brain-tumor/statistics: :text=A%20person’s%20likelihood%20of
%20developing,nervous%20system%20(CNS)%20tumors.
[^2]: Kasraeian, Sina et al. “A comparison of fine-needle aspiration, core biopsy, and surgical biopsy in the diagnosis of extremity soft tissue masses.”
*Clinical orthopaedics and related research* vol. 468,11 (2010): 2992-3002. doi:10.1007/s11999-010-1401-x
