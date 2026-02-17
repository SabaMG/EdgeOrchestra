## 🎯 Vision du Projet

**Nom de code :** EdgeOrchestra

**Objectif :** Créer une infrastructure open-source de federated learning et edge computing qui transforme des devices Apple inutilisés en cluster de calcul ML distribué.

## 📚 Phase 0 : Research & State of the Art (Semaine 1)

**Papers essentiels à lire :**
1. **Federated Learning:**
   - "Communication-Efficient Learning of Deep Networks from Decentralized Data" (Google, 2017) - le paper fondateur
   - "Federated Learning: Challenges, Methods, and Future Directions" (2019)
   - "FedAvg vs FedProx" - comparaison des algorithmes

2. **Edge Computing:**
   - "Edge Intelligence: Paving the Last Mile of AI with Edge Computing" (2019)
   - "In-Edge AI: Intelligentizing Mobile Edge Computing" (2020)

3. **Mobile ML:**
   - "MLPerf Mobile Inference Benchmark" - pour comprendre les perfs devices
   - Apple's Core ML performance papers

**Solutions existantes à analyser :**
- Flower (framework federated learning)
- TensorFlow Federated
- PySyft
- FedML

**Ton angle de différenciation :**
- Focus sur devices Apple (optimisations Metal/Core ML)
- Zero-config orchestration (plug & play)
- Battery-aware scheduling (crucial pour mobile)
- Hybrid edge-cloud (ton Hetzner comme fallback)

## 🏗️ Architecture Technique

```
┌─────────────────────────────────────────────────────┐
│                    MacBook (Dev)                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Model Registry & Training Orchestrator     │   │
│  │  - Push models                               │   │
│  │  - Define federated tasks                    │   │
│  │  - Aggregate results                         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│           Raspberry Pi (Orchestrator)               │
│  ┌──────────────────────────────────────────────┐   │
│  │  - Device registry & health monitoring      │   │
│  │  - Task scheduler (battery/CPU aware)       │   │
│  │  - Model distribution                       │   │
│  │  - Gradient aggregation (FedAvg/FedProx)    │   │
│  │  - Communication coordinator                │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Redis: Task queue & state management       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
          │              │              │
          ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ iPhone  │    │  iPad   │    │ iPhone  │
    │  Node   │    │  Node   │    │  Node   │
    ├─────────┤    ├─────────┤    ├─────────┤
    │ Worker  │    │ Worker  │    │ Worker  │
    │ Agent   │    │ Agent   │    │ Agent   │
    │         │    │         │    │         │
    │ Local   │    │ Local   │    │ Local   │
    │ Training│    │ Training│    │ Training│
    │         │    │         │    │         │
    │ Battery │    │ Battery │    │ Battery │
    │ Monitor │    │ Monitor │    │ Monitor │
    └─────────┘    └─────────┘    └─────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Hetzner (Optional) │
              │  - Cloud fallback   │
              │  - Model storage    │
              │  - Metrics DB       │
              └─────────────────────┘
```

## 🛠️ Tech Stack

**Raspberry Pi (Orchestrator):**
- Python 3.11+
- FastAPI (API REST pour communication)
- Redis (task queue & state)
- PostgreSQL (metrics, historique)
- Docker (pour faciliter deployment)
- Prometheus + Grafana (monitoring)

**iPhone/iPad (Workers):**
- Swift + SwiftUI (app native)
- Core ML (inference optimisée)
- MLX ou TensorFlow Lite (training on-device)
- Background Tasks framework (training async)
- Network framework (communication efficace)

**MacBook (Control Plane):**
- Python (CLI tool)
- PyTorch (définition & export modèles)
- Web dashboard (React/Next.js)

**Communication:**
- gRPC (efficace, bi-directionnel)
- Protocol Buffers (sérialisation)
- mDNS/Bonjour (découverte automatique devices)

## 📋 Roadmap Détaillé

### **Phase 1 : Foundation (Semaines 1-2)**

**Semaine 1 : Setup & Discovery**
- [ ] Commander Raspberry Pi 4 (8GB RAM recommandé) + accessories
- [ ] Setup Raspberry Pi : OS, Docker, PostgreSQL, Redis
- [ ] Implémenter device discovery protocol (mDNS/Bonjour)
- [ ] Créer structure projet (monorepo recommended)
- [ ] Définir protocole de communication (Protocol Buffers schemas)

**Semaine 2 : Basic Communication**
- [ ] Serveur gRPC sur Raspberry Pi
- [ ] App iOS basique qui se connecte et s'enregistre
- [ ] Heartbeat system (devices ping orchestrator toutes les 30s)
- [ ] Device registry avec métadonnées (model, iOS version, battery, etc.)
- [ ] Dashboard web basique (liste des devices connectés)

### **Phase 2 : Model Distribution (Semaines 3-4)**

**Semaine 3 : Model Management**
- [ ] Model registry sur Raspberry Pi
- [ ] API pour upload modèles depuis Mac (format Core ML)
- [ ] Système de versioning de modèles
- [ ] Compression de modèles pour transmission efficace
- [ ] Cache local sur devices

**Semaine 4 : Inference Distribuée**
- [ ] Téléchargement & installation de modèles sur iOS
- [ ] Exécution d'inférence avec Core ML
- [ ] Envoi des résultats à l'orchestrateur
- [ ] Load balancing basique (round-robin)
- [ ] Metrics : latence, throughput, accuracy

### **Phase 3 : Federated Learning Core (Semaines 5-7)**

**Semaine 5 : Local Training**
- [ ] Implémenter training on-device (TensorFlow Lite ou MLX)
- [ ] Data loading depuis stockage local
- [ ] Gradient computation
- [ ] Test avec modèle simple (MNIST ou CIFAR-10 pour commencer)

**Semaine 6 : Federated Averaging**
- [ ] Implémentation FedAvg sur orchestrateur
- [ ] Agrégation de gradients de multiples devices
- [ ] Update du modèle global
- [ ] Re-distribution aux clients
- [ ] Tests avec 2-3 devices simultanés

**Semaine 7 : Optimisations**
- [ ] Compression de gradients (quantization, sparsification)
- [ ] Differential privacy (ajout de bruit aux gradients)
- [ ] Secure aggregation (optionnel, cryptographie)
- [ ] Tests de convergence

### **Phase 4 : Battery & Resource Awareness (Semaines 8-9)**

**Semaine 8 : Smart Scheduling**
- [ ] Battery level monitoring sur iOS
- [ ] Thermal state monitoring
- [ ] Scheduler qui priorise devices avec >50% batterie
- [ ] Pause training si batterie <20%
- [ ] Background task scheduling (training pendant charge nocturne)

**Semaine 9 : Adaptive Learning**
- [ ] Profiling de perfs par device (temps/epoch, consommation)
- [ ] Adaptation taille batch selon device
- [ ] Prédiction temps restant training
- [ ] Auto-scaling : plus de rounds si devices disponibles

### **Phase 5 : Advanced Features (Semaines 10-12)**

**Semaine 10 : Hybrid Edge-Cloud**
- [ ] Intégration Hetzner comme compute node additionnel
- [ ] Fallback automatique si pas assez de devices edge
- [ ] Cost optimization (edge first, cloud si nécessaire)
- [ ] Benchmark edge vs cloud (latence, coût, énergie)

**Semaine 11 : Advanced FL Algorithms**
- [ ] Implémentation FedProx (gère better heterogeneous devices)
- [ ] FedNova (normalisation pour convergence)
- [ ] Comparaison empirique FedAvg vs FedProx vs FedNova

**Semaine 12 : Production Ready**
- [ ] Error handling robuste (device disconnect, crash, etc.)
- [ ] Checkpointing & recovery
- [ ] Logging structuré
- [ ] Documentation API
- [ ] Tests unitaires & intégration

## 📊 Expérimentations & Metrics

**Use Cases à implémenter :**

1. **Image Classification (CIFAR-10)**
   - Dataset distribué sur devices
   - Mesure convergence vs centralized training
   - Impact nombre de devices sur accuracy finale

2. **Keyboard Prediction (Next-Word)**
   - Chaque device a typing patterns différents
   - Federated learning pour modèle global
   - Privacy-preserving (pas de partage de texte)

3. **Anomaly Detection**
   - Chaque device détecte patterns locaux
   - Modèle global apprend de tous
   - Use case : détection d'activité inhabituelle

**Metrics à tracker :**
- **Performance ML :**
  - Accuracy vs rounds
  - Loss convergence
  - Time to convergence
  - Comparison centralized vs federated

- **System :**
  - Communication overhead (MB transmitted/round)
  - Latency per round
  - Energy consumption per device
  - Device participation rate

- **Scalability :**
  - Performance avec 1, 2, 3+ devices
  - Impact device heterogeneity
  - Stragglers handling

## 📝 Deliverables Recherche

**Paper Structure (à écrire parallèlement) :**

1. **Introduction**
   - Motivation : recycling old devices for ML
   - Challenges : battery, heterogeneity, communication

2. **Related Work**
   - Federated learning frameworks
   - Edge ML systems
   - Mobile ML optimization

3. **System Design**
   - Architecture détaillée
   - Protocol design
   - Scheduling algorithms

4. **Implementation**
   - Tech stack choices & justifications
   - Challenges rencontrés
   - Optimizations

5. **Evaluation**
   - Expériences sur 3 use cases
   - Comparaisons avec baselines
   - Ablation studies

6. **Discussion**
   - Limitations
   - Future work
   - Real-world applicability

**Où soumettre :**
- Conférences : MLSys, MobiCom, EdgeSys
- Workshops : FL-NeurIPS, TinyML Summit
- Journals : ACM TECS, IEEE IoT Journal

## 💡 Innovations Potentielles (Differentiate ta recherche)

1. **Battery-Aware Federated Learning**
   - Algorithme qui balance convergence speed vs energy
   - Peut devenir une contribution novel

2. **Heterogeneity-Robust Aggregation**
   - iPhone 12 vs iPhone 7 ont perfs très différentes
   - Weighted aggregation selon device capability

3. **Opportunistic Training**
   - Learn from usage patterns (quand user charge device)
   - Maximize training sans impacter UX

4. **Privacy Metrics**
   - Quantifier privacy preservation
   - Trade-off utility vs privacy

## 🚀 Quick Wins pour Portfolio

**Demo Videos à faire :**
1. "Zero-config setup : plug devices, they auto-discover"
2. "Live dashboard showing federated training in action"
3. "Battery drops, system pauses gracefully"
4. "Convergence comparison: federated vs centralized"

**GitHub Repo Structure :**
```
edge-orchestra/
├── orchestrator/        # Raspberry Pi code
├── ios-worker/         # iOS app
├── control-plane/      # Mac CLI tool
├── dashboard/          # Web monitoring
├── experiments/        # Jupyter notebooks avec résultats
├── papers/            # LaTeX draft
└── docs/              # Documentation
```

## 🎓 Bonus : Lien avec ton Stage

**Angles à mentionner en entretien :**
- "J'ai implémenté un système de federated learning from scratch"
- "J'ai géré l'hétérogénéité des devices (key challenge en FL)"
- "J'ai optimisé pour contraintes mobiles (batterie, compute)"
- "J'ai comparé empiriquement différents algorithmes FL"
- "J'ai écrit un paper technique sur mes findings"

**Questions qu'on te posera probablement :**
- Pourquoi federated learning vs centralized ?
- Comment gérer stragglers (slow devices) ?
- Communication efficiency : combien de MB par round ?
- Privacy guarantees : differential privacy implementation ?

---

## 🤔 Questions pour toi avant de commencer :

1. **Tu veux que je détaille plus une phase en particulier ?** (ex: la partie iOS app, l'algorithme FedAvg, le monitoring, etc.)

2. **Tu as déjà une idée du premier use case à implémenter ?** (je recommande MNIST pour commencer, c'est simple)

3. **Tu veux qu'on planifie les milestones de façon plus granulaire ?** (ex: objectifs semaine par semaine avec checklist précise)

4. **Niveau hardware : tu veux commander le Raspberry Pi maintenant ou tu veux prototyper d'abord sans ?** (tu peux commencer avec juste iPhone + Mac pour tester la comm)

5. **Tu veux que je te fasse un starter code pour un composant en particulier ?** (ex: le gRPC server, l'app iOS basique, le FedAvg implementation)

Dis-moi ce qui t'aiderait le plus et on plonge dans les détails ! 🚀