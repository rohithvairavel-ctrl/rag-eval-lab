# Feature Stores
A feature store centralizes feature definitions, serving (online/offline), and point-in-time correctness for training.
It prevents training-serving skew by using the same transforms. Offline store for batch training; online for low-latency inference.
Versioning and lineage support auditability. Popular options include Feast and vendor platforms.
Without a store, teams often reimplement features inconsistently across notebooks and services.
