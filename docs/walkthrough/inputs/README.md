# Walkthrough input images

Everything in the walkthrough is fictional. The faces are **computer-rendered people who do not
exist**; the names, phone numbers and places are invented.

| File | What it is |
|---|---|
| `reference_2017.png`, `reference_2025.png` | Two renders of synthetic identity #2 (DigiFace-1M, images 6 and 18), upscaled 3x. They play the two reference photos of the "missing person". |
| `sighting_1_platform.jpg`, `sighting_2_bus_stand.jpg`, `sighting_3_market.jpg` | Collages made for the demo: six synthetic faces each (the same identity #2, in other renders with other accessories and lighting, plus five other synthetic people), with a fake camera header and timestamp. They stand in for CCTV or phone stills of a crowd. |

## Source and licence

Faces: **DigiFace-1M** (Microsoft Research), 1M+ synthetic face images for face recognition,
downloaded from the Hugging Face mirror
[`lhoestq/digiface1m_720k`](https://huggingface.co/datasets/lhoestq/digiface1m_720k)
(official page: <https://github.com/microsoft/DigiFace1M>).

- Licence: **Research Use of Data Agreement v1.0 (R-UDA)**: non-commercial research use only;
  redistribution is allowed if the credit below is kept and recipients are bound to the same
  terms. By using these images you agree to the R-UDA
  (<https://github.com/microsoft/DigiFace1M/blob/main/LICENSE>). Do not use them in a product.
- Disclaimer from the authors: some rendered faces may resemble real people by chance.
- Citation: Bae, de La Gorce, Baltrusaitis, Hewitt, Chen, Valentin, Cipolla, Shen.
  *DigiFace-1M: 1 Million Digital Face Images for Face Recognition.* WACV 2023.

The collages were composed by this project and are not part of DigiFace-1M. Rebuilding them needs
the dataset; the recorded screenshots and video do not.
