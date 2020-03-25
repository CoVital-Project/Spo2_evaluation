# SPO2 Evaluation

Welcome aboard the SpO2 evaluation project to help relief of the Covid2019 pandemic.

## Goals

1) Build a dataset enabling us to evaluate and develop algorithms for spo2 estimation within 2% accuracy from a smartphone camera images.
2) Compare latest state-of-the-art method for spo2 estimation from smartphone camera images.
3) Develop and ship a method to estimate spo2 within 2% accuracy from a smartphone camera images.

## Outcomes

1) A dataset from evaluating and developing spo2 estimation algorithms.
	* The dataset should be available publicly to enable anyone to implement and test their algorithm.
2) A decision wether to ship a given algorithm (and why) or implement our own.

## What is the plan?

1) Decide how to analyze the results of the algorithm and which data to compile.

Again the method is describe in the [overleaf document](https://www.overleaf.com/read/kwfmchzmmgtm).
Reach out again with suggestions and improvements. The current method is implemented in [the spo2 repo](https://github.com/CoVital-Project/Spo2_evaluation)

2) Compile a dataset from algorithm evaluation with the necessary data. 

The technical specification of the dataset are described [here](https://www.overleaf.com/read/kwfmchzmmgtm). 
If you would like to help with the specification, reach out to me and I can give you access to the overleaf document.

3) URGENT: consent forms

I think we're at that point where we need someone to make sure we can get that data and that we won't be in legal problem over it. If someone has an office in their University or company that could check that out for us, it would be great, please be in touch! I'm thinking GDPR.

4) Implement the comparison and some baseline methods

@Yoni and I are reimplementing some algorithms for testing. Anyone is welcome to pitch in and add their own implementation (be it personal or from a paper). I'll add a list of implemented method and the reference paper in the root README of the project. One thing we need to decide is the output format of the algorithm to run the tests.

## Where are we at

We've done step 1 (but it's all flexible, if you have input on them go for it, nothing is set in stone) and are moving forward on step 4.

Step 2 is starting and we need someone to get onto step 3.

# Quick description of the packages:

Each package has a README describing the use.


# Paper reimplemented

> @INPROCEEDINGS{7145228, 
> author={F. {Lamonaca} and D. L. {Carnì} and D. {Grimaldi} and A. {Nastro} and M. {Riccio} and V. {Spagnolo}}, 
> booktitle={2015 IEEE International Symposium on Medical Measurements and Applications (MeMeA) Proceedings}, 
> title={Blood oxygen saturation measurement by smartphone camera}, 
> year={2015}, 
> volume={}, 
> number={}, 
> pages={359-364},} 


> @article{NEMCOVA2020101928,
> title = "Monitoring of heart rate, blood oxygen saturation, and blood pressure using a smartphone",
> journal = "Biomedical Signal Processing and Control",
> volume = "59",
> pages = "101928",
> year = "2020",
> issn = "1746-8094",
> doi = "https://doi.org/10.1016/j.bspc.2020.101928",
> url = "http://www.sciencedirect.com/science/article/pii/S1746809420300847",
> author = "Andrea Nemcova and Ivana Jordanova and Martin Varecka and Radovan Smisek and Lucie Marsanova and Lukas Smital > and Martin Vitek",
> keywords = "Heart rate, Blood oxygen saturation, Blood pressure, Photoplethysmogram, Health monitoring, Smartphone, Android application",
> }

> @INPROCEEDINGS{6959086, author={A. K. {Kanva} and C. J. {Sharma} and S. {Deb}}, booktitle={Proceedings of The 2014 International Conference on Control, Instrumentation, Energy and Communication (CIEC)}, title={Determination of SpO2 and heart-rate using smartphone camera}, year={2014}, volume={}, number={}, pages={237-241},} 

# Paper to review/reimplement

> @INPROCEEDINGS{8037323, author={E. J. {Wang} and W. {Li} and J. {Zhu} and R. {Rana} and S. N. {Patel}}, booktitle={2017 39th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)}, title={Noninvasive hemoglobin measurement using unmodified smartphone camera and white flash}, year={2017}, volume={}, number={}, pages={2333-2336},} 

> @inproceedings{10.1145/2971648.2971653,
> author = {Wang, Edward Jay and Li, William and Hawkins, Doug and Gernsheimer, Terry and Norby-Slycord, Colette and Patel, Shwetak N.},
> title = {HemaApp: Noninvasive Blood Screening of Hemoglobin Using Smartphone Cameras},
> year = {2016},
> isbn = {9781450344616},
> publisher = {Association for Computing Machinery},
> address = {New York, NY, USA},
> url = {https://doi.org/10.1145/2971648.2971653},
> doi = {10.1145/2971648.2971653},
> booktitle = {Proceedings of the 2016 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
> pages = {593–604},
> numpages = {12},
> keywords = {anemia, hemoglobin, photoplethysmography, camera, mobile health, blood screening},
> location = {Heidelberg, Germany},
> series = {UbiComp ’16}
> }
