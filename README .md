# **🧠 RaSed — AI & Machine Learning Engine**

## **🎯 Project Overview**

This repository hosts the **Brain of the RaSed Safety System**. While the hardware node (Edge) collects raw environmental data, this ML Engine processes that data in real-time to transition the system from "reactive threshold logic" to "proactive AI-driven prediction."

Our model analyzes multi-modal telemetry (Gas, Temperature, Dust) to classify the industrial environment into three states: **SAFE**, **SMOKE**, or **FIRE**.

## **📐 System Architecture**

The **RaSed** architecture follows a three-tier model designed for high-availability industrial safety:

1. **Sensing Layer (Data Acquisition):** \- Deployed multi-sensor arrays (BME280, MQ8, MQ3, PMS5003, IR Array) capture real-time chemical and physical telemetry.  
2. **Edge Computing Node (Pico W):** \- Acts as the execution tier, aggregating sensor data and publishing structured **JSON** packets via **MQTT**. It handles time-critical actuation locally (relays, servos) to ensure sub-second response times.  
3. **Cloud & AI Brain (Analytical Tier):** \- The centralized intelligence layer. It subscribes to the **MQTT Broker**, processes incoming telemetry through the **Random Forest** model, and issues high-level commands back to the edge node.

## **🌲 Why Random Forest?**

We selected the **Random Forest Classifier** for the RaSed engine for several critical industrial reasons:

* **Robustness to Overfitting:** In industrial environments, sensor data is often noisy. Random Forest, being an ensemble of decision trees, generalizes better than single-tree models, reducing the risk of false alarms.  
* **Handling Multi-modal Data:** Our input features (Gas, Temperature, Dust) have different scales and units. Random Forest naturally handles non-linear relationships and diverse feature types without requiring complex data normalization.  
* **Feature Importance:** It provides clear insights into which sensors contribute most to a "FIRE" classification, helping us refine our hardware deployment strategy.  
* **Real-time Efficiency:** Once trained, the inference time for Random Forest is extremely low, making it ideal for the sub-second response requirements of our safety system.

## **🛠️ Tech Stack**

* **Language:** Python 3.9+  
* **Core ML Library:** Scikit-Learn (Random Forest Classifier).  
* **Data Handling:** Pandas & NumPy for telemetry normalization.  
* **Communication:** Paho-MQTT for real-time edge-to-cloud synchronization.

## **📊 Dataset & Features**

The model is trained on a comprehensive industrial dataset generated through controlled environment simulation and real-world testing. The dataset includes:

* **Input Features:**  
  * temp (Celsius): Ambient temperature range (20°C \- 80°C).  
  * pm2\_5 (Particulate Matter): 0 to 500 ![][image1].  
  * tvoc (Total VOCs): Measured by CCS811 sensor (0 \- 32768 ppb).  
  * eco2 (Equivalent CO2): 400 to 8192 ppm.  
  * h2 (Hydrogen gas): Raw analog values from MQ8 (reflecting leak concentrations).  
  * mq3 (Alcohol vapor): Raw analog values from MQ3.  
* **Target Labels (Classes):**  
  * 0: **SAFE** (Normal environmental parameters).  
  * 1: **SMOKE** (Elevated particulate and VOC levels detected).  
  * 2: **FIRE** (Significant temperature rise, gas leakage, and IR flame signal convergence).

**Developed as part of the RaSed Smart Safety Management System — Graduation Project 2026**

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAaCAYAAADrCT9ZAAADs0lEQVR4Xu1WTUhUURR+gwZFQVrZkDPjnRlXUtRi+qEoiGhRUS1aVCBt+iGKFkGEEiFtXESLJBKisoKQqCzc1CYXRpBS0K4fgkBDigw3QkJU2vf57tXjcd44T4vpZz74uO+ee+655953zrnX84r4+5FKpaLGmHUVFRXz9Ni/hkgymayvqqo6BGaw6W7KtFLBUF1dvRgOngFn67HpABtcAvbSJvvYdB3/tlIrGErh3FVwrR6YKbDhMtjdBL7SYwUDnKkB7/6OPMOG18D2QbBNjxUMcOYYqeW/ErBfi7BequUFAZxpTSQSy7R8JrBR0xyLxeLsY7M7SK0XhBJMXMhWD+QC8yefIgRHbqApVeISFjK7biSTycxi0SH57ZRonwWK+uNT/Q1C/gPjG9nH9668awQUD4BDegKMnc1W+SBPQrcTiz5G+xTdJm4qaPOQ71WiCOY1Yk472n7wBPgCbAG7oP8WY2njp8IT8DLYkU6n5zsD/IbsGngU3AY+p12xRiC4eCvY58KDwILlNBKNRudKZYTmSsg/g6fRLTH+yQ6DD7Jt2NpJSRnDm2vyD6Pt5Px4PD7H6vNO/QJ+gL16z/5Z9EfArdIOx+izCfPwqKysXIQJL7XDxs+RAalLo9bB1+7Pw/lVxo+OU1LXATa3eOrkITsM/Vr7SnqH/jM3JuzdkaHNDYfJ0UDAyHoY+6YdpkNcRMoYmpSBFz27Cev8MOxslroWEcjPayGjhptxa6Otc2NuXWx8p5wD2QBYI2XTAR8EbSYgnMFO5gqdE393EH9mhdM1WaLDgQcEWw1abjFpbbmuDFEejD0UHmC5F7K4jkGEs17A5dEF8Ar6aW8813uNXzVHYfzoaHR9gZyvKxfORhyWWHeCPfrBvEeex6BzPe981RDh3ENjlNnq12H8nNmNttmzV4rxC9QgFt/APpxebgLC2Uzxuso3nF1kWb9OUseNhYbNvxHwO9hl/GvhEZzYjvYh+AabWi2msCrzqvgEnfeYfxvfH20ETIDVC3xdYWw/+NWICICdS+j3GVXVsc5xjLWjbZKFLDSMzSFcDwk+AHhNeOP5MfowkPoaNiUm5S+vGMhvmhxFho5Db4EnKjjtgGVCbQzugaLloWByFBwN/GkDvT3yXmZYypAUcuZiizf5dVVYwKmhbA5r2PzpNuIp53KdB6HUueEGVmgtLzhMQMHRsCF6D+zHRo6gbQR7wPta1x7ELVcE/yhgs/tCFIEI84hFjFfEFNX3nDfTfCuiiCL+e/wEbhgKLWRDvNQAAAAASUVORK5CYII=>
