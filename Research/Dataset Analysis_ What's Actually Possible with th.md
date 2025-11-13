<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Dataset Analysis: What's Actually Possible with the Toyota GR Cup Data

Based on comprehensive analysis of the hackathon resources and dataset documentation, here's what you can actually build with the provided data:

### **Available Datasets (VIR Race 1 as Example)**[^1][^2]

**Time Series Data:**

- `vir_lap_time_R1.csv`: Total lap duration for each car/lap
- `vir_lap_start_R1.csv`: Timestamp when lap started
- `vir_lap_end_R1.csv`: Timestamp when lap finished
- `R1_vir_telemetry_data.csv`: High-frequency telemetry measurements

**Race Results:**

- `05_Results by Class GR Cup Race 1 Official_Anonymized.CSV`: Final standings with positions, total laps, elapsed time, gaps to competitors
- `03_Provisional Results_Race 1_Anonymized.CSV`: Pre-adjustment results
- `05_Provisional Results by Class_Race 1_Anonymized.CSV`: Class-based standings (mostly Amateur class)

**Performance Analysis:**

- `23_AnalysisEnduranceWithSections_Race_1_anonymized.CSV`: Section-by-section lap breakdowns (track divided into segments)


### **Telemetry Data Fields**[^1]

The most valuable dataset contains:

- **Speed**: Actual vehicle speed (likely in mph or km/h)
- **Gear**: Current gear selection (1-6, sequential transmission)
- **ath**: Throttle blade position (engine throttle opening)
- **aps**: Throttle pedal position (driver input)
- **pbrake_r**: Rear brake pressure
- **pbrake_f**: Front brake pressure
- **Timestamp**: Likely included for synchronization


### **What's Possible vs. Not Possible**

#### ✅ **HIGHLY FEASIBLE with Available Data**

**1. Counterfactual Strategy Analysis** *(YOUR TOP RECOMMENDATION)*

- **Data supports:** Lap times, section times, race positions, gaps to competitors
- **Analysis possible:**
    - Build causal models linking section performance → lap times → race outcomes
    - Simulate alternative pit strategies using historical timing patterns
    - Answer "what-if" questions: "If driver X had improved section 3 by 0.5s, how would final position change?"
- **Limitation:** No explicit pit stop data visible, but you can infer strategy from lap time anomalies (slow laps = pit stops)

**2. Real-Time Anomaly Detection** *(YOUR SECONDARY RECOMMENDATION)*

- **Data supports:** High-frequency telemetry (speed, throttle, brake), lap times across multiple laps
- **Analysis possible:**
    - Train LSTM/Transformer models on "normal" racing patterns per track section[^3][^4]
    - Detect mechanical issues (brake pressure anomalies, throttle inconsistencies)
    - Flag suboptimal driver inputs (late braking, early throttle lift)
    - Identify strategic opportunities (unusual competitor slow-downs)
- **Strong fit:** Telemetry data with speed/gear/throttle/brake is perfect for time-series anomaly detection[^5][^3]

**3. Driver Performance Profiling** *(ALREADY DONE - See MTP DNA Analyzer)*

- **Data supports:** Multi-lap consistency, section-by-section performance, best lap analysis
- **Analysis possible:**
    - Cluster drivers by style (aggressive vs. smooth, qualifying vs. race pace)[^6]
    - Identify strengths/weaknesses by track section[^3]
    - Compare throttle/brake application patterns across drivers
- **Note:** The MTP DNA Analyzer from 2024 already did this, so you'd need significant innovation to differentiate[^6]

**4. Hybrid Physics-ML Tire Degradation Model**

- **Data supports:** Lap time progression over race distance, speed telemetry
- **Analysis possible:**
    - Model lap time degradation as proxy for tire wear
    - Use speed through corners + lap number to estimate grip loss
    - Physics-informed NN: encode known tire degradation curves, learn residuals from data[^7][^8]
- **Limitation:** No explicit tire compound, temperature, or pressure data visible - you'll need to infer degradation from lap time trends

**5. Section-by-Section Optimization**

- **Data supports:** `23_AnalysisEnduranceWithSections` provides track segment breakdowns[^1]
- **Analysis possible:**
    - Identify which sections drivers gain/lose most time[^9][^10][^11]
    - Calculate "par times" for each section based on top performers[^10][^11]
    - Visualize driver improvement opportunities per section
    - Compare throttle/brake patterns in specific corners using telemetry[^12][^3]

**6. Racing Line Optimization via Telemetry**

- **Data supports:** Speed + gear + throttle/brake at high frequency
- **Analysis possible:**
    - Reconstruct approximate racing lines using speed profiles[^13][^14]
    - Compare corner entry/exit speeds between drivers
    - Identify optimal shift points and braking zones[^15][^3]
    - Detect understeer/oversteer patterns from speed scrubbing[^15]


#### ⚠️ **PARTIALLY FEASIBLE (Requires Workarounds)**

**7. Pre-Event Prediction Models**

- **Data supports:** Historical race results, lap times, section times across multiple races
- **Analysis possible:** Train models to predict finishing position, fastest lap, top 3
- **Limitation:** You need multiple race files (VIR, other tracks) to build robust predictions. Check if hackathon provides multiple race datasets
- **Workaround:** Use transfer learning from other racing series or synthesize additional training data

**8. Multi-Agent Racing Simulation**

- **Data supports:** Race results show multi-car interaction (gaps, positions)
- **Analysis possible:** Build game-theoretic models of overtaking scenarios[^16][^17]
- **Limitation:** No GPS position data or car-to-car proximity metrics visible
- **Workaround:** Infer interactions from gap changes in section timing data

**9. Graph Neural Networks for Race Dynamics**

- **Data supports:** Position changes, gaps between drivers over time
- **Analysis possible:** Model race as temporal graph with drivers as nodes[^18]
- **Limitation:** Limited spatial relationship data without GPS/track position
- **Workaround:** Create graph edges based on time gaps and position changes in section timing


#### ❌ **NOT FEASIBLE (Missing Critical Data)**

**10. Real-Time Pit Strategy Optimization**

- **Missing:** Explicit pit stop times, tire compound choices, fuel consumption
- **Cannot:** Build pit window optimizer without knowing when cars actually pitted
- **Workaround:** Infer pit laps from anomalous lap times, but this is imprecise

**11. Track Position / GPS-Based Analysis**

- **Missing:** GPS coordinates, XY positions on track
- **Cannot:** Generate racing line visualizations, overtaking zone heatmaps
- **Workaround:** Approximate positions using speed curves + section boundaries

**12. Weather/Track Condition Impact**

- **Missing:** Weather data, track temperature, grip levels
- **Cannot:** Model performance changes due to environmental factors
- **Note:** If you have access to external weather APIs for race dates/locations, you could enrich the dataset

**13. Suspension/Aero Telemetry**

- **Missing:** Ride height, downforce, suspension travel
- **Cannot:** Analyze setup optimization or mechanical grip issues
- **Available alternative:** Use speed/brake/throttle patterns as proxies for setup balance


### **Optimal Project Given Dataset Constraints**

**Recommended: "RaceIQ - Counterfactual Strategy Engine with Real-Time Anomaly Detection"**

This hybrid system is **optimally suited** to the available data:

**Component 1: Counterfactual Strategy Analyzer**

- Uses: Lap times, section times, race positions, gaps
- Techniques: Causal inference (DoWhy), SHAP explainability, Monte Carlo simulation
- Output: Interactive dashboard answering "what-if" scenarios about section improvements, alternative strategies

**Component 2: Real-Time Anomaly Detection**

- Uses: Telemetry (speed, gear, throttle, brake), lap-by-lap timing
- Techniques: LSTM/Transformer for time-series, VMD decomposition, z-score anomalies[^4]
- Output: Alert system flagging unusual patterns (mechanical issues, driver errors, strategic opportunities)

**Integration:**

- When anomaly detected → automatically trigger counterfactual analysis
- Example: "Anomaly: Driver X's brake pressure 15% lower in Sector 3 → Counterfactual: If corrected, estimated 0.8s lap time improvement → Projected to move from P5 to P3"

**Why This Works:**

1. **Data completeness:** All required fields are present in the datasets[^1]
2. **Novel approach:** Neither technique widely used in GR Cup context[^19][^20][^4]
3. **Practical value:** Directly actionable for race engineers[^21][^22]
4. **Technical depth:** Demonstrates mastery of causal inference + deep learning
5. **Hackathon-friendly:** Both components can be built independently and integrated later

### **Alternative Strong Options**

**Option 2: "SectorMaster - Section-by-Section Driver Training Tool"**

- Focus on `23_AnalysisEnduranceWithSections` data[^1]
- Create interactive heatmaps showing driver strengths/weaknesses per section[^9][^10]
- Compare telemetry patterns (throttle/brake) in each section[^3]
- Generate personalized training recommendations
- **Best for:** "Driver Training \& Insights" category

**Option 3: "DegradePredict - Hybrid Physics-ML Tire Life Forecaster"**

- Model lap time degradation over race distance
- Physics-informed neural networks for tire wear[^23][^7]
- Uncertainty quantification with Bayesian methods[^24]
- Predict optimal stint lengths and pit windows
- **Best for:** "Pre-Event Prediction" category

Would you like me to provide implementation details, Python pseudocode, or library recommendations for the counterfactual + anomaly detection hybrid system?
<span style="display:none">[^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54]</span>

<div align="center">⁂</div>

[^1]: https://hackthetrack.devpost.com/resources

[^2]: https://hackthetrack.devpost.com/rules

[^3]: https://www.rpm-rush.com/mastering-advanced-sim-racing-telemetry-analysis/

[^4]: https://www.nature.com/articles/s41598-024-72982-z

[^5]: https://www.tinybird.co/blog/real-time-anomaly-detection

[^6]: https://www.youtube.com/watch?v=iHxYWBuUcWQ

[^7]: https://www.monolithai.com/blog/tyre-degradation-self-learning-models

[^8]: https://people.ece.cornell.edu/land/courses/ece5760/FinalProjects/f2020/sn438_fs383_rs872/sn438_fs383_rs872/index.html

[^9]: https://www.racingtv.com/news/james-willoughby-why-sectional-timing-data-matters-in-racing

[^10]: https://www.geegeez.co.uk/sectional-timing-and-how-to-use-it/

[^11]: https://raceiq.com/par-sectionals-fsp/

[^12]: https://www.youtube.com/watch?v=KtrU8xnnKiM

[^13]: https://dspace.mit.edu/bitstream/handle/1721.1/64669/706825301-MIT.pdf

[^14]: https://www.shellecomarathon.com/2025-programme/regional-asia-pacific-and-the-middle-east/_jcr_content/root/main/section_1633744779/call_to_action_1681610291/links/item0.stream/1740730639827/2231cc55fa45e1c94ef0fad92684e680b2ccfe08/apme-2025-ota-data-and-telemetry-batavia-gasoline-team.pdf

[^15]: http://blog.axisofoversteer.com/2012/09/VIRtrackguide.html

[^16]: https://repositories.lib.utexas.edu/items/7fb6b86a-3e2c-4df5-89f0-d29a3be28165

[^17]: https://proceedings.mlr.press/v229/werner23a/werner23a.pdf

[^18]: https://arxiv.org/html/2307.03759v3

[^19]: https://arxiv.org/html/2501.04068v1

[^20]: https://arxiv.org/html/2505.13324v1

[^21]: https://aws.amazon.com/blogs/media/real-time-storytelling-the-aws-architecture-behind-formula-1-track-pulse/

[^22]: https://reelmind.ai/blog/f1-race-today-ai-powered-race-analysis-and-strategy-breakdown

[^23]: https://arious.uk/ai/advanced-modeling-of-tire-degradation-for-predictive-maintenance-f1

[^24]: https://github.com/Aishwarya4823/Formula-E-Racing-Lap-Prediction-Using-Machine-Learning

[^25]: https://future-of-data-hackathon-2025.devpost.com

[^26]: https://hack4dev.org/newmodel/

[^27]: https://github.com/PGEHackathon/data

[^28]: https://asmedigitalcollection.asme.org/mechanicaldesign/article/147/4/044506/1212561/Presenting-Hackathon-Data-for-Design-Research-A

[^29]: https://treehacks-2025.devpost.com

[^30]: https://www.poliruralplus.eu/knowledge-transfer/blog/data-to-develop-hackathon-co-creating-the-future-of-regional-development-through-ai-and-collaboration/

[^31]: https://hackthetrack.devpost.com

[^32]: https://www.hackthetrack.org/2025-event

[^33]: https://community.openai.com/t/what-are-you-building-2025-projects-hackathon-thread/1243270?page=4

[^34]: https://hackthefest.com

[^35]: https://www.sciencedirect.com/science/article/pii/S2352340924003019

[^36]: https://pressroom.toyota.com/toyota-gr-cup-leverages-cutting-edge-digital-trophies-to-enhance-driver-and-fan-engagement/

[^37]: https://cros.ec.europa.eu/2025EuropeanBigDataHackathon

[^38]: https://www.youtube.com/watch?v=wLCCFaYK4yI

[^39]: https://communities.sas.com/t5/SAS-Communities-Library/SAS-Hackathon-2025-Individual-Student-Track-Participant-Guide/ta-p/973929

[^40]: https://www.linkedin.com/posts/devpost_hack-the-track-presented-by-toyota-gr-activity-7391152754783723520-irVh

[^41]: https://ches.iacr.org/2025/challenge.php

[^42]: https://www.reddit.com/r/AUTOMOBILISTA/comments/1ipcbvj/any_way_to_get_my_throttle_and_brake_telemetry/

[^43]: https://www.facebook.com/groups/647165782874898/posts/1626083278316472/

[^44]: https://app.tracktitan.io/track-guides/en/honda_civic_type_r_1997-virginia_international_raceway_full-forza-Track-Guide/8

[^45]: https://www.youtube.com/watch?v=ZrMU1XyVpeA

[^46]: https://www.reddit.com/r/F1Technical/comments/xjgpke/how_are_the_interval_times_calculated_during_the/

[^47]: https://www.simracingsystem.com/showthread.php?tid=9629\&action=lastpost

[^48]: https://g87.bimmerpost.com/forums/showthread.php?t=2055904

[^49]: https://www.youtube.com/watch?v=ve6OmvR6bWQ

[^50]: https://www.raceresult.com/en/support/kb?id=34519-Lap-Race---Time-Limit

[^51]: https://app.tracktitan.io/track-guides/en/toyota_gr86-virginia_2022_full-iRacing-Track-Guide/3

[^52]: https://www.reddit.com/r/F1Technical/comments/v4yaa3/how_are_sectors_determined_on_a_formula_1_track/

[^53]: https://www.facebook.com/groups/racebox/posts/2586984405023516/

[^54]: https://www.instagram.com/reel/DQnR94yDxDh/

