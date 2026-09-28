import React from 'react';
import { Layers, Scan, Brain, Film, Sliders, Laptop } from 'lucide-react';

export const About: React.FC = () => {
  return (
    <div className="page-container max-w-4xl mx-auto">
      <div className="section-header text-center mb-8">
        <h1>Project Architecture & Technical Methodology</h1>
        <p className="text-muted">
          Detailed technical documentation for the BTech Final Year Project: "Deepfake Detection in Image and Video Using ResNet18-MTCNN Pipeline"
        </p>
      </div>

      <div className="about-sections-stack">
        {/* Section 1: Problem Overview */}
        <section className="card mb-6">
          <div className="about-card-header">
            <Layers className="text-primary mr-2" size={24} />
            <h2>1. What Are Deepfakes?</h2>
          </div>
          <p>
            Deepfakes are synthetic media generated using deep neural networks—such as Generative Adversarial Networks (GANs),
            Variational Autoencoders (VAEs), and modern Diffusion models—to replace or manipulate the facial likeness or expressions
            of an individual. These manipulations pose significant challenges to identity verification, digital forensics,
            and information authenticity.
          </p>
        </section>

        {/* Section 2: MTCNN Face Localization */}
        <section className="card mb-6">
          <div className="about-card-header">
            <Scan className="text-primary mr-2" size={24} />
            <h2>2. MTCNN Facial Localization & 20% Padding</h2>
          </div>
          <p>
            Rather than feeding full arbitrary images to the classifier, this system utilizes <strong>MTCNN (Multi-task Cascaded Convolutional Networks)</strong>
            to accurately detect and isolate facial regions.
          </p>
          <ul className="bullet-list">
            <li><strong>Minimum Face Size:</strong> 20 pixels</li>
            <li><strong>Detection Thresholds:</strong> [0.6, 0.7, 0.7] across P-Net, R-Net, and O-Net stages.</li>
            <li>
              <strong>Largest Face Selection:</strong> When multiple individuals appear, the face with the maximum bounding box
              area <code>(w × h)</code> is automatically chosen.
            </li>
            <li>
              <strong>20% Face Padding:</strong> The bounding box is expanded by 20% along width and height
              (<code>pad_x = floor(w * 0.2)</code>, <code>pad_y = floor(h * 0.2)</code>) and clamped to image boundaries. This ensures
              forensic facial boundary blending artifacts around jawlines and foreheads are captured.
            </li>
            <li><strong>Standard Size:</strong> Resized to <code>224 × 224</code> pixels for ResNet18 input.</li>
          </ul>
        </section>

        {/* Section 3: ResNet18 Transfer Learning */}
        <section className="card mb-6">
          <div className="about-card-header">
            <Brain className="text-primary mr-2" size={24} />
            <h2>3. ResNet18 Architecture & Transfer Learning</h2>
          </div>
          <p>
            The feature extraction backbone is based on <strong>ResNet18</strong> pre-trained on ImageNet. Pre-training gives the
            network rich perceptual filters for edges, textures, and anatomical structures.
          </p>
          <div className="pipeline-flow-box">
            <code>Input (3 × 224 × 224) → ResNet18 Backbone → 512-dim Feature Vector → Dropout(0.5) → Linear(512 → 1) → Single Logit</code>
          </div>
          <ul className="bullet-list">
            <li><strong>Single Output Logit:</strong> The model outputs a raw unbounded scalar logit.</li>
            <li><strong>Loss Function:</strong> Trained with <code>BCEWithLogitsLoss(logit, target)</code> for optimal numerical stability.</li>
            <li><strong>Inference Activation:</strong> <code>probability = sigmoid(logit)</code> yields a calibrated fake probability in [0.0, 1.0].</li>
          </ul>
        </section>

        {/* Section 4: Video Processing Pipeline */}
        <section className="card mb-6">
          <div className="about-card-header">
            <Film className="text-primary mr-2" size={24} />
            <h2>4. Video Frame Sampling & Aggregation</h2>
          </div>
          <p>
            Videos often contain thousands of redundant frames. To optimize local laptop performance, the system uses OpenCV to
            sample up to <strong>64 evenly spaced frames</strong> across the total duration.
          </p>
          <p>
            For each sampled frame, MTCNN locates the face and ResNet18 predicts an individual frame score (p_i). Frames lacking a detectable face are safely skipped.
            The final video probability is calculated via arithmetic mean aggregation:
          </p>
          <div className="math-box">
            <code>P_video = (1 / N) &times; &Sigma; p_i &nbsp; (for all N valid face frames)</code>
          </div>
        </section>

        {/* Section 5: Decision Thresholds */}
        <section className="card mb-6">
          <div className="about-card-header">
            <Sliders className="text-primary mr-2" size={24} />
            <h2>5. Three-State Classification Thresholds</h2>
          </div>
          <p>
            Binary classification models can produce borderline predictions near 0.50. To prevent false absolutes in a student demonstration,
            we configure a three-state decision boundary:
          </p>
          <div className="threshold-table-wrapper">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Condition</th>
                  <th>Classification Label</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>Score &lt; 0.40</code></td>
                  <td><span className="badge badge-success">REAL</span></td>
                  <td>Strong indication of authentic, unmanipulated imagery.</td>
                </tr>
                <tr>
                  <td><code>0.40 ≤ Score ≤ 0.60</code></td>
                  <td><span className="badge badge-warning">UNCERTAIN</span></td>
                  <td>Ambiguous or low-confidence region requiring human review.</td>
                </tr>
                <tr>
                  <td><code>Score &gt; 0.60</code></td>
                  <td><span className="badge badge-danger">FAKE</span></td>
                  <td>Strong indication of facial manipulation or deepfake generation.</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-muted mt-2">
            * Thresholds are fully configurable in <code>backend/.env</code> via <code>LOWER_THRESHOLD</code> and <code>UPPER_THRESHOLD</code>.
          </p>
        </section>

        {/* Section 6: Local Technology Stack */}
        <section className="card mb-6">
          <div className="about-card-header">
            <Laptop className="text-primary mr-2" size={24} />
            <h2>6. System Implementation Stack</h2>
          </div>
          <div className="tech-stack-grid">
            <div className="tech-stack-item">
              <strong>Frontend:</strong> React 18, Vite, TypeScript, Axios
            </div>
            <div className="tech-stack-item">
              <strong>Backend:</strong> FastAPI, Uvicorn, Python 3.9+, Pydantic
            </div>
            <div className="tech-stack-item">
              <strong>Database:</strong> MongoDB Local Community Server (Motor driver)
            </div>
            <div className="tech-stack-item">
              <strong>Computer Vision:</strong> OpenCV, Pillow, facenet-pytorch (MTCNN)
            </div>
            <div className="tech-stack-item">
              <strong>Machine Learning:</strong> PyTorch 2.x, torchvision (ResNet18)
            </div>
            <div className="tech-stack-item">
              <strong>Concurrency:</strong> FastAPI BackgroundTasks (no Redis/Celery required)
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
