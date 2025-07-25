import React, { useState, useEffect } from 'react';

import {
  Button,
  MenuItem,
  Select,
  TextField,
  Typography,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { LineChart } from '@mui/x-charts/LineChart';
import { CircularProgress } from '@mui/material';

type LayerConfig = { neurons: number };

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [model, setModel] = useState('linear');
  const [layers, setLayers] = useState<LayerConfig[]>([{ neurons: 8 }]);
  const [epochs, setEpochs] = useState(10);
  const [gamma, setGamma] = useState(0.05);
  const [C, setC] = useState(1.0);
  const [learningRate, setLearningRate] = useState(0.01);
  const [activationType, setActivationType] = useState(0);      // 0 = ReLU, 1 = Tanh
  const [batchSize, setBatchSize] = useState(32);               // Taille des mini-batches
  const [lambdaL2, setLambdaL2] = useState(0.0);                // Coefficient L2
const neurons = layers[0]?.neurons || 8; // Un seul nombre

  const [sigma, setSigma] = useState(1.0);
  const [selectedGraph, setSelectedGraph] = useState<string[]>([]);

  const [modelName, setModelName] = useState('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isTrained, setIsTrained] = useState(false);

  const [loadedModelName, setLoadedModelName] = useState<string | null>(null);

  const [resultsFile, setResultsFile] = useState<{
    results: { name: string; data: number[] }[];
  }>({ results: [] });
  const [resultsTraining, setResultsTraining] = useState<{ files: string[] }>({
    files: [],
  });
  const [prediction, setPrediction] = useState<string | null>(null);


  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImage(URL.createObjectURL(file));
    setImageFile(file);
    setPrediction(null);
  };

  const handleAddLayer = () => {
    setLayers([...layers, { neurons: 8 }]);
  };

  const handleChangeNeurons = (index: number, value: number) => {
    const updated = [...layers];
    updated[index].neurons = value;
    setLayers(updated);
  };

  const imageToFloatArray = (file: File): Promise<number[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
          const canvas = document.createElement('canvas');
          canvas.width = 32;
          canvas.height = 32;
          const ctx = canvas.getContext('2d');
          if (!ctx) return reject('Erreur contexte canvas');

          ctx.drawImage(img, 0, 0, 32, 32);
          const imageData = ctx.getImageData(0, 0,32, 32);
          const data = imageData.data;
          const grayPixels: number[] = [];

          for (let i = 0; i < data.length; i += 4) {
            const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
            grayPixels.push(gray / 255);
          }

          resolve(grayPixels);
        };
        img.src = e.target?.result as string;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleTrain = async () => {
    setIsTraining(true); 
    try {
      const labels = ['A', 'B', 'C'];
      const n_features = 32 * 32;

      const createUrl = {
        linear: 'linear/create/',
        mlp: 'mlp/create/',
        mlp_deep: 'mlp_deep/create/',
        rbf: 'rbfn_multiclass/create/',
        svm: 'svm/create/',
      }[model];

      const createBody =
        model === 'linear'
          ? {
              n_features,
              learning_rate: learningRate,
              max_epochs: epochs,
              activation_type: activationType,
              n_classes: labels.length,
            }
          : model === 'mlp'
          ? {
              n_inputs: n_features,
              n_hidden: layers[0]?.neurons || 8,
              n_classes: labels.length,
              learning_rate: learningRate,
              epochs,
            }
          : model === 'rbf'
          ? {
              sigma,
              learning_rate: learningRate,
              epochs,
              n_classes: labels.length,
            }
          : model === 'svm'
          ? {
              gamma,
              c: C,
              learning_rate: learningRate,
              epochs,
            }
            : model === 'mlp_deep'
            ? {
                 n_features,
                  hidden_units: neurons,
                  n_layers: layers.length,
                  n_classes: labels.length,
                  learning_rate: learningRate,
                  epochs,
                  activation_type: activationType,
                  batch_size: batchSize,
                  lambda_: lambdaL2
              }

          : {};

      await fetch(`http://localhost:8000/${createUrl}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(createBody),
      });


      const trainConfig: any =
  model === 'mlp_deep'
    ? {
        labels,
        n_features,
        hidden_units: neurons,
        n_layers: layers.length,
        n_classes: labels.length,
        learning_rate: learningRate,
        epochs,
        activation_type: activationType,
        batch_size: batchSize,
        lambda_: lambdaL2
      }
    : model === 'mlp'
    ? {
        labels,
        learning_rate: learningRate,
        epochs,
        activation_type: activationType,
        n_hidden: layers[0]?.neurons || 8
      }
    : {
        labels,
        learning_rate: learningRate,
        epochs,
        sigma,
        gamma,
        c: C
      };



      const res = await fetch(`http://localhost:8000/${model}/train/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainConfig),
      });


      if (!res.ok) throw new Error(`Erreur entraînement (${model})`);

      const result = await res.json();
      const mseLoss = result.mse_loss || 0.0;

      const runName = `run_${model}_${Date.now()}`;
      const lossArray = result.loss_per_epoch || Array(epochs).fill(parseFloat(mseLoss.toFixed(4)));

      setResultsTraining((prev) => ({
        files: [...prev.files, runName],
      }));

      setResultsFile((prev) => ({
        results: [...prev.results, { name: runName, data: lossArray }],
      }));

      setSelectedGraph((prev) => [...prev, runName]);

      alert(`${model.toUpperCase()} entrainé avec succès (Loss: ${mseLoss.toFixed(4)})`);
      setIsTrained(true);

    } catch (err: any) {
      alert(`${err.message}`);
      setIsTrained(false);
    } finally {
      setIsTraining(false); 
    }
  };

  const handlePredict = async () => {
    if (!imageFile) return alert('Veuillez importer une image.');

    try {
      const pixels = await imageToFloatArray(imageFile);

      if (pixels.length !== 32 * 32) {
        return alert("l’image doit faire 64x64 (4096 pixels)");
      }
      const input = model === 'linear' ? { x: [pixels] } : { x: pixels };

      let endpoint = '';
      if (model === 'linear') {
        endpoint = 'linear/predict/';
      } else if (model === 'mlp') {
        endpoint = 'mlp/predict/';
      } else if (model === 'rbf') {
        endpoint = 'rbfn/predict/';
      } else if (model === 'svm') {
        endpoint = 'svm/predict/';
       } else if (model === 'mlp_deep') {
        endpoint = 'mlp_deep/predict/';
      } else {
        return alert('Modèle non pris en charge pour la prédiction');
      }

      const res = await fetch(`http://localhost:8000/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      });

      if (!res.ok) throw new Error('Erreur lors de la prédiction');

      const data = await res.json();

      if (Array.isArray(data.predictions)) {
        const predictionObj = data.predictions[0];
        if (predictionObj && typeof predictionObj === 'object') {
          setPrediction(predictionObj.label || '?');
        } else {
          setPrediction('?');
        }
      } else if (typeof data.label === 'string') {
        setPrediction(data.label);
      } else {
        setPrediction('?');
      }


    } catch (err: any) {
      alert(`${err.message}`);
    }
  };

  const handleSaveModel = async () => {
    if (!modelName.trim()) return alert('Please enter a model name');
    try {
      const res = await fetch('http://localhost:8000/model/save/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName.trim() }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Unknown error');
      }

      alert('Model saved successfully.');
      fetchModelList();
    } catch (err: any) {
      alert(`Error saving model: ${err.message}`);
    }
  };


  const handleLoadModel = async (name: string) => {
    try {
      const res = await fetch('http://localhost:8000/model/load/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, model_type: model }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Erreur inconnue');
      }

      const data = await res.json();
      if (data.type) setModel(data.type);
      setLoadedModelName(name);
      alert(`Modèle "${name}" chargé avec succès`);
    } catch (err: any) {
      alert(`Erreur chargement modèle : ${err.message}`);
    }
  };

  const handleImportModelFromFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const modelName = file.name;
    setModelName(modelName);
    try {
      await handleLoadModel(modelName);
    } catch (err: any) {
      alert(`Erreur import modèle : ${err.message}`);
    }
  };

  const fetchModelList = async () => {
    const res = await fetch('http://localhost:8000/model/list/');
    const data = await res.json();
    setAvailableModels(data.models || []);
  };
  useEffect(() => {
    fetchModelList();
  }, []);

    


  return (
    <div style={{ display: 'flex', height: '150vh' }}>
      {/* Sidebar */}
      <div style={{ width: '300px', background: '#f0f0f0', padding: '1rem' }}>
        <Typography variant="h6">🧠 Modele</Typography>
        <Select
          fullWidth
          value={model}
          onChange={(e) => {
            setModel(e.target.value);
            setLayers([{ neurons: 8 }]);
          }}
        >
          <MenuItem value="linear">Linear</MenuItem>
          <MenuItem value="mlp">MLP</MenuItem>
          <MenuItem value="mlp_deep">MLP Deep</MenuItem>
          <MenuItem value="rbf">RBF</MenuItem>
          <MenuItem value="svm">SVM</MenuItem>

        </Select>

        {(model === 'mlp') && (
          <div style={{ marginTop: '1rem' }}>
            <Typography variant="subtitle1">📐 Couches</Typography>
            {layers.map((layer, index) => (
              <TextField
                key={index}
                label={`Neurones couche ${index + 1}`}
                type="number"
                value={layer.neurons}
                fullWidth
                sx={{ marginBottom: '0.5rem' }}
                onChange={(e) => {
                  const value = parseInt(e.target.value);
                  if (!isNaN(value)) handleChangeNeurons(index, value);
                }}
              />
            ))}
            <Button
                      variant="outlined"
                      onClick={handleAddLayer}
                      fullWidth
                      sx={{ marginTop: '0.5rem' }}
                    >
                      Add layer
                    </Button>
                  </div>
                )}
        {model === "mlp_deep" && (
        <>
          <Typography variant="h6" sx={{ mt: 2 }}>⚙️ Configuration MLP Deep</Typography>

          <TextField
            fullWidth
            label="Batch size"
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            sx={{ mt: 2 }}
          />

          <TextField
            fullWidth
            label="Lambda L2"
            type="number"
            value={lambdaL2}
            onChange={(e) => setLambdaL2(parseFloat(e.target.value))}
            sx={{ mt: 2 }}
          />

          <Typography variant="subtitle1" sx={{ mt: 2 }}>🔢 Couches cachées</Typography>
          {layers.map((layer, idx) => (
            <TextField
              key={idx}
              fullWidth
              label={`Neurones couche ${idx + 1}`}
              type="number"
              value={layer.neurons}
              onChange={(e) => {
                const updated = [...layers];
                updated[idx].neurons = parseInt(e.target.value);
                setLayers(updated);
              }}
              sx={{ mt: 1 }}
            />
          ))}

          <Button
            variant="outlined"
            onClick={() => setLayers([...layers, { neurons: 64 }])}
            sx={{ mt: 1 }}
          >
            ➕ Ajouter une couche
          </Button>

          <Typography variant="subtitle1" sx={{ mt: 2 }}>🔌 Fonction d’activation</Typography>
          <Select
            fullWidth
            value={activationType}
            onChange={(e) => setActivationType(Number(e.target.value))}
            sx={{ mb: 2 }}
          >
            <MenuItem value={0}>ReLU</MenuItem>
            <MenuItem value={1}>Tanh</MenuItem>
          </Select>
        </>
      )}



        {model === 'linear' && (
          <>
            <Typography variant="subtitle1" sx={{ marginTop: '1rem' }}>
              🔧 Config Linear
            </Typography>
            <TextField
              label="Learning Rate"
              type="number"
              fullWidth
              value={learningRate}
              inputProps={{ step: 0.001 }}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              sx={{ marginBottom: '0.5rem' }}
            />
           
          </>
        )}

        {(model === 'mlp' || model === 'rbf') && (
          <TextField
            label="Learning Rate"
            type="number"
            fullWidth
            value={learningRate}
            inputProps={{ step: 0.001 }}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            sx={{ marginTop: '1rem', marginBottom: '0.5rem' }}
          />
        )}

        {model === 'rbf' && (
          <TextField
            label="Sigma"
            type="number"
            fullWidth
            value={sigma}
            inputProps={{ step: 0.1 }}
            onChange={(e) => setSigma(parseFloat(e.target.value))}
            sx={{ marginBottom: '0.5rem' }}
          />
        )}

        {model === 'svm' && (
        <>
          <TextField
            label="Gamma"
            type="number"
            fullWidth
            value={gamma}
            inputProps={{ step: 0.01 }}
            onChange={(e) => setGamma(parseFloat(e.target.value))}
            sx={{ marginTop: '1rem', marginBottom: '0.5rem' }}
          />
          <TextField
            label="C (Penalty)"
            type="number"
            fullWidth
            value={C}
            inputProps={{ step: 0.1 }}
            onChange={(e) => setC(parseFloat(e.target.value))}
            sx={{ marginBottom: '0.5rem' }}
          />
          <TextField
            label="Learning Rate"
            type="number"
            fullWidth
            value={learningRate}
            inputProps={{ step: 0.001 }}
            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            sx={{ marginBottom: '0.5rem' }}
          />
        </>
      )}


        <Typography variant="h6" sx={{ marginTop: '2rem' }}>
          🔁 Epochs
        </Typography>
        <TextField
          type="number"
          fullWidth
          value={epochs}
          onChange={(e) => {
            const value = parseInt(e.target.value);
            if (!isNaN(value)) setEpochs(value);
          }}
          inputProps={{ min: 1 }}
        />

      <Button
      variant="contained"
      fullWidth
      sx={{
        marginTop: '1.5rem',
        backgroundColor: isTrained ? 'green' : undefined,
        '&:hover': {
          backgroundColor: isTrained ? '#2e7d32' : undefined,
        },
      }}
      onClick={handleTrain}
      disabled={isTraining}
    >
      {isTraining ? (
        <>
          <CircularProgress size={18} sx={{ color: 'white', marginRight: 1 }} />
          Training...
        </>
      ) : isTrained ? (
        'Model trained'
      ) : (
        'Start training'
      )}
    </Button>


        <Typography variant="h6" sx={{ marginTop: '2rem' }}>💾 Model Save / Load</Typography>

<TextField
  label="Model name"
  fullWidth
  value={modelName}
  onChange={(e) => setModelName(e.target.value)}
  sx={{ marginBottom: '0.5rem' }}
/>

<Button
  variant="contained"
  fullWidth
  onClick={handleSaveModel}
  sx={{ marginBottom: '1rem' }}
>
  💾 Save model
</Button>

<Typography variant="subtitle1">📂 Available Models</Typography>
{availableModels.map((name) => (
  <Button
    key={name}
    variant={loadedModelName === name ? "contained" : "outlined"}
    fullWidth
    sx={{
      marginTop: '0.25rem',
      backgroundColor: loadedModelName === name ? 'green' : undefined,
      color: loadedModelName === name ? 'white' : undefined,
      '&:hover': {
        backgroundColor: loadedModelName === name ? '#2e7d32' : undefined,
      },
    }}
    onClick={() => handleLoadModel(name)}
  >
    📥 Load {name}
  </Button>
))}

<Button
  component="label"
  variant="contained" 
  fullWidth
 sx={{ mt: 2, mb: 0.5 }} 
>
  📤 Load model from file
  <input
    type="file"
    accept=".model"
    hidden
    onChange={handleImportModelFromFile}
  />
</Button>

      </div>


      {/* Right zone */}
      <div style={{ flex: 1, padding: '2rem' }}>
        <Typography variant="h5" gutterBottom>
          Select image for prediction
        </Typography>

        <Button
          component="label"
          variant="contained"
          startIcon={<CloudUploadIcon />}
          sx={{ marginBottom: '1rem' }}
        >
          Upload an image
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={handleImageUpload}
          />
        </Button>

        {image && (
          <>
            <img
              src={image}
              alt="Image importée"
              style={{ maxWidth: '300px', display: 'block', borderRadius: 8 }}
            />
            <Button
              variant="contained"
              color="secondary"
              sx={{ marginTop: '1rem' }}
              onClick={handlePredict}
            >
              Predict
            </Button>
            {prediction && (
              <Typography variant="h6" sx={{ marginTop: '1rem' }}>
                Prediction : {prediction}
              </Typography>
            )}
          </>
        )}

        <div className="view_training_results" style={{ marginTop: '3rem' }}>
          <Typography variant="h5">Training results</Typography>

          <div className="container-results" style={{ marginBottom: '1rem' }}>
            {resultsTraining?.files.map((file, index) => (
              <div key={index}>
                <Button
                  variant="contained"
                  sx={{
                    background: '#2874a6',
                    color: '#fff',
                    marginTop: '10px',
                    marginRight: '10px',
                  }}
                  onClick={() => {
                    setSelectedGraph((prev) =>
                      prev.includes(file)
                        ? prev.filter((f) => f !== file)
                        : [...prev, file]
                    );
                  }}
                >
                  {file}
                </Button>
              </div>
            ))}
          </div>

          <div className="container-charts">
            <LineChart
              xAxis={[
                {
                  data: Array.from(
                    {
                      length: Math.max(
                        ...(resultsFile?.results.map((r) => r.data.length) || [
                          1,
                        ])
                      ),
                    },
                    (_, i) => i + 1
                  ),
                  label: 'Epochs',
                },
              ]}
              series={
                resultsFile?.results
                  .filter((r) => selectedGraph.includes(r.name))
                  .map((r) => ({
                    data: r.data,
                    label: r.name,
                  })) || []
              }
              width={900}
              height={400}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
