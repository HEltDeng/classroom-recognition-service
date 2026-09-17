# A5 integration

Added immutable `A5-OMR-v1` and `A5-OCR-v1` templates (1480 x 2100 canonical coordinates; 148 x 210 mm). Their JSON geometry is identical to the Web PDF renderer's copies. OMR has 40 rectangular choice slots; OCR has ten text slots. Slot numbers are local to the page; only the main backend maps them to exam questions.

`POST /recognize/page` accepts matching `template_version` and `page_type=OMR|OCR`. The response echoes both. Existing `A-v0.1` stays supported. No answer keys, students or scores enter the recognition engine.

OMR v0.2 estimates paper brightness around each choice box. The prior within-box 90th percentile misclassified a uniformly black fill as blank. A dedicated solid-black regression test now covers this defect; original blank/multiple/light/ambiguous tests remain.

`OCR_CPU_THREADS` defaults to 2 and can be set to 1 on a shared CPU host. The deployed model selection is `PP-OCRv5_mobile_rec`; the existing server model remains supported locally.

Validation: eleven tests; actual generated PDF with synthetic marks recognized 40/40 choices. Ten printed English word crops recognized by the real local PaddleOCR provider. These are synthetic/printed-text results, not real-camera or handwriting accuracy evidence.
