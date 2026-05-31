Based on your team's specific situation (**3 people, 11-week development cycle, NVIDIA RTX 4090 cloud servers required**), and aligned with the technical red lines and delivery requirements of the project "Basketball Ball-Handler Re-Identification Algorithm Development in Complex Motion Scenes," I prepared a standard software engineering budget plan.

We use **Bottom-Up Estimating**, mapped to your previously refined 6 WBS phases for cost estimation.

## I. Team Profile
- Topic: Basketball Ball-Handler Re-Identification Algorithm Development in Complex Motion Scenes
- Team size: 3 people
- Development cycle: 11 weeks
- Equipment needs: Rent NVIDIA RTX 4090 cloud servers

We adopt the bottom-up method for project budgeting.

## II. Core Budget Formula

According to software engineering standards, the total budget model is:

$$\text{Total Budget} = \text{Labor Cost} + \text{Compute Rental} + \text{Data and Materials} + \text{Indirect Management Cost} + \text{Contingency Reserve}$$

## III. Detailed Budget Breakdown and Estimation

### 1. Labor Cost Estimation (Allocated)

As this is a competition and R&D project, we estimate part-time input for a 3-person team over 11 weeks, using average salaries for junior-to-mid algorithm/engineering roles as a reference.

* **Staffing and responsibilities**:
	- **Ye Li** (Project Manager): Overall project planning and risk management; design TransReID model improvement strategies; formulate training and tuning plans; review model performance and technical documents.
	- **Chengyou Long** (Project Member): Build the development environment and deploy the baseline model; write and debug model code; complete model training and inference; optimize model inference speed to meet competition requirements.
	- **Siyu Lei** (Project Member): Complete dataset preprocessing and management; conduct model testing and metric reporting; compile all project submission materials; back up project assets; organize development logs.

* **Effort calculation**: 11 weeks $\times$ 5 workdays = 55 person-days per person. Total **165 person-days** for 3 people.
* **Rate**: Average cost about 500 RMB per person-day.

$$\text{Labor Cost} = 165 \text{ person-days} \times 500 \text{ RMB/person-day} = 82,500 \text{ RMB}$$

### 2. Hardware and Compute Rental Budget

The project requires benchmarking and stress testing on **NVIDIA RTX 4090**. We use cloud compute (AutoDL) for 11 weeks, including intensive training, multiple rounds of hyperparameter search, and ablation studies.

* **Intensive training (W2-W7, 6 weeks)**: 2x 4090 servers at high load, 2.5 RMB/hour, 16 hours per day per server:

$$6 \text{ weeks} \times 7 \text{ days} \times 16 \text{ hours} \times 2.5 \text{ RMB} \times 2 \text{ servers} = 3,360 \text{ RMB}$$

* **Daily debugging and engineering stress tests (W1, W8-W11, 5 weeks)**: 1x 4090 server, 8 hours per day:

$$5 \text{ weeks} \times 7 \text{ days} \times 8 \text{ hours} \times 2.5 \text{ RMB} \times 1 \text{ server} = 700 \text{ RMB}$$

* **Cloud storage and data transfer**: Reserve 440 RMB for dataset expansion and backup.

$$\text{Compute Rental Total} = 3,360 + 700 + 440 = 4,500 \text{ RMB}$$

### 3. Data and Third-Party Services

* **Annotation calibration**: The task includes cluttered indoor/outdoor backgrounds and same-jersey teammates. WBS includes "hard-sample cleaning and annotation calibration." Budget **1,500 RMB** for part-time help or paid tools to refine bounding boxes for blurred/occluded samples.

* **External open-source dataset retrieval and compliance**: Allocate **500 RMB** for sourcing and compliant downloading of external datasets.

### 4. Deliverables and Indirect Costs

* **Packaging of deliverables**: Layout of the detailed technical plan document, visual design for the presentation deck, and HD recording/editing of the demo video. Reserve **1,500 RMB**.

* **Daily operating overhead**: Communication, bandwidth, and meeting costs over 11 weeks. Fixed at **2,000 RMB**.

### 5. Contingency Reserve

* AI R&D has large metric uncertainty (e.g., pushing mAP from 91.1% to $\ge 91.5\%$ can hit bottlenecks) and strict latency red lines of $\le 40\text{ms}/30\text{ms}$. We typically reserve **10%** of total cost for extreme tuning and schedule risk.

$$\text{Contingency Reserve} = 92,500 \times 10\% \approx 9,250 \text{ RMB}$$

---

## IV. Overall Budget Sheet

Below is a standardized budget table suitable for project application or financial reporting:

| Level-1 Category | Level-2 Category | Basis / Description | Estimated Amount (RMB) | Share of Total |
| --- | --- | --- | --- | --- |
| **1. Labor** | R&D service cost | 3 people $\times$ 11 weeks $\times$ 5 days = 165 person-days, 500 RMB/person-day | 82,500 | 81.08% |
| **2. Compute and Equipment** | Cloud 4090 rental | Core period 2 servers $\times$ 6 weeks + debug period 1 server $\times$ 5 weeks + storage expansion | 4,500 | 4.42% |
| **3. Data and Services** | Data cleaning and calibration | Manual review and micro-annotation for similar jerseys and heavy occlusions | 2,000 | 1.97% |
| **4. Indirect and Deliverables** | Overhead and packaging | 11-week ops overhead + high-quality video/plan/PPT polishing | 3,500 | 3.44% |
| **5. Risk Reserve** | R&D contingency | Emergency R&D if accuracy not met or latency exceeds $>40\text{ms}$ | 9,250 | 9.09% |
| **Total Project Budget** |  | **Grand Total (one hundred one thousand seven hundred fifty RMB)** | **101,750** | **100%** |
