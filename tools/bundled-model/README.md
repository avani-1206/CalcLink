# Bundled CalcInk symbol model

How `models/calcink_symbols.onnx` was made (no ML framework needed, just NumPy + OpenCV):

```bash
pip install numpy opencv-python onnxruntime
python synth.py            # writes samples.png: a grid of synthetic training symbols
python train.py 4000 12    # 4000 samples/class/epoch, 12 epochs (~15 min on 2 CPU cores)
cp calcink_symbols.onnx ../../models/
```

- `synth.py`: stroke templates for `0-9 + - x ÷ = ( ) y`, randomly rotated, sheared, stretched, wobbled and drawn with random pen widths. Rendering matches `rasterize()` in `src/app/recognizer.js`.
- `train.py`: a small CNN written in NumPy (im2col convolutions, Adam), fresh synthetic data every epoch.
- `onnx_writer.py`: writes the ONNX protobuf directly (≈100 lines). The output matches onnxruntime to within 1e-7.
