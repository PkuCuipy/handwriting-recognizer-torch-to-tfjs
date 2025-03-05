import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as d3 from 'd3';

const HandwritingRecognitionApp = () => {
  const modelRef = useRef(null);
  const [modelReady, setmodelReady] = useState(false);

  const [predictions, setPredictions] = useState(Array(10).fill(1/10));

  const [brushSize, setBrushSize] = useState(2);
  const [isDrawing, setIsDrawing] = useState(false);

  const canvasRef = useRef(null);
  const gridSize = 28;
  const pixelSize = 10;
  const canvasSize = gridSize * pixelSize;

  const chartRef = useRef(null);


  // Load the deep model for MNIST digit recognition
  useEffect(() => {
    async function loadModel() {
      try {
        modelRef.current = await tf.loadGraphModel('model.json');
        setmodelReady(true);
        console.log("Model loaded successfully!");

        // Make one prediction to warm up the model,
        // otherwise the first draw will be laggy
        predict();

      } catch (error) {
        console.error("Failed to load model", error);
      }
    }

    loadModel();
  }, []);


  // Initialize the canvas to white at the beginning
  useEffect(() => {
    clearCanvas();
    console.log("Canvas initialized to pure white");
  }, []);


  // Update the bar-chart when predictions change
  useEffect(() => {
    if (!predictions) return;

    const margin = { top: 30, right: 10, bottom: 30, left: 30 };
    const width = 300 - margin.left - margin.right;
    const height = 180 - margin.top - margin.bottom;

    d3.select(chartRef.current).selectAll("*").remove();

    const svg = d3.select(chartRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
      .range([0, width])
      .domain(Array.from({ length: 10 }, (_, i) => i.toString()))
      .padding(0.2);

    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x));

    const y = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    svg.append("g")
      .call(d3.axisLeft(y).ticks(5));

    svg.selectAll("bars")
      .data(predictions)
      .enter()
      .append("rect")
      .attr("x", (d, i) => x(i.toString()))
      .attr("y", d => y(d))
      .attr("width", x.bandwidth())
      .attr("height", d => height - y(d))
      .attr("fill", (d, i) => {
        const maxIndex = predictions.indexOf(Math.max(...predictions));
        return i === maxIndex ? "#4CAF50" : "#9E9E9E";
      });
  }, [predictions]);


  // Make predictions based on the current canvas
  const predict = async () => {
    console.log("Predicting with model", modelRef.current);
    if (!modelRef.current) return;

    // Get the pixel data from the canvas
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvasSize, canvasSize);

    // Sample the canvas to get a 28x28 pixel image as model input
    const inputData = new Float32Array(gridSize * gridSize);
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        // Use the center of each "big pixel" as the pixel value
        const sampleIndex = ((y * pixelSize + Math.floor(pixelSize / 2)) * canvasSize + (x * pixelSize + Math.floor(pixelSize / 2))) * 4;
        const value = 1 - imageData.data[sampleIndex] / 255;   // only one channel is needed, also remember to reverse the color
        inputData[y * gridSize + x] = value;
      }
    }

    const input = tf.tensor(inputData).reshape([1, gridSize, gridSize, 1]);   // -> [1, 28, 28, 1]
    const output = modelRef.current.predict(input);
    const probabilities = tf.softmax(output).dataSync();  // use softmax to transform "logits" into "probabilities"

    setPredictions(Array.from(probabilities));    // convert to JS array and update state

    // Clean up the tensors to avoid memory leaks
    input.dispose();
    output.dispose();
  };


  // Set the flag to true when the user starts drawing.
  // Only necessary to support mouses (distinguish between hovering and actually drawing. on touch screen, there's no "hovering")
  const startDrawing = (e) => {
    console.log("startDrawing()");
    setIsDrawing(true);
    draw(e);
  };


  // Draw on the canvas
  const draw = (e) => {
    console.log("draw(), isDrawing:", isDrawing);

    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Get the mouse/touch position
    if (e.clientX === undefined) {  // For touch events, use the first finger's position
      e.clientX = e.touches[0].clientX;
      e.clientY = e.touches[0].clientY;
    }
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Map to a position within the 28x28 grid
    const gridX = Math.floor(x / pixelSize);
    const gridY = Math.floor(y / pixelSize);

    // Draw a "big-pixel" (a square) with the selected brush size
    ctx.fillStyle = '#000000';
    for (let offsetY = -brushSize + 1; offsetY < brushSize; offsetY++) {
      for (let offsetX = -brushSize + 1; offsetX < brushSize; offsetX++) {
        const currentGridX = gridX + offsetX;
        const currentGridY = gridY + offsetY;

        const withinGrid = currentGridX >= 0 && currentGridX < gridSize && currentGridY >= 0 && currentGridY < gridSize;
        if (!withinGrid) {
          continue;
        }

        const distance = Math.sqrt(offsetX * offsetX + offsetY * offsetY);
        if (distance > brushSize) {
          continue;
        }

        // Opacity based on the distance from the center of the brush: the further away, the more transparent
        const opacity = 1 - (distance / brushSize);
        ctx.globalAlpha = opacity;
        ctx.fillRect(
          currentGridX * pixelSize,
          currentGridY * pixelSize,
          pixelSize,
          pixelSize
        );
      }
    }

    // Reset the opacity
    ctx.globalAlpha = 1;

    // Make a prediction every time the user draws a stroke
    predict();
  };

  const stopDrawing = () => {
    console.log("Stop Drawing");
    setIsDrawing(false);
  };


  // Clear the canvas to white
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    // Reset the predictions by making a prediction on a blank canvas
    predict();
  };

  return (
    <div className="flex flex-col items-center p-4 gap-4 w-full max-w-sm mx-auto touch-none">
      {!modelReady && (
        <div className="text-lg">Loading Model...</div>
      )}
      <>

        {/* Control panel */}
        <div className="w-full flex items-center justify-between bg-gray-50 p-4 rounded-lg border border-gray-200">

          {/* Brush size slider */}
          <div className="flex items-center">
            <span className="mr-2">Brush Size</span>
            <input
              type="range"
              min="1"
              max="3"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              className="w-24"
            />
            <span className="ml-2">{brushSize}</span>
          </div>

          {/* Clear button */}
          <button
            onClick={clearCanvas}
            className="bg-red-500 text-white px-3 py-1 rounded-full"
          >
            Clear All
          </button>
        </div>


        {/* Canvas */}
        <div className="border border-gray-800 rounded-lg overflow-hidden">
          <canvas
            ref={canvasRef}
            width={canvasSize}
            height={canvasSize}

            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}

            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
        </div>


        {/* Bar Chart */}
        <div className="w-full max-w-xs flex flex-col items-center justify-between bg-gray-50 p-2 rounded-lg border border-gray-200">
          <div className="w-full">
            <div ref={chartRef} className="w-full flex justify-center"/>
          </div>
        </div>

        {/* Text Result */}
        <div className="w-full text-lg max-w-xs flex flex-col items-center justify-between bg-gray-50 p-2 rounded-lg border border-gray-200">
          <div className="">
            <span className="text-xl">Guess&nbsp;</span>
            <span className="font-bold text-green-600 text-2xl">{predictions.indexOf(Math.max(...predictions))}</span>
            <span className="ml-2">with probability {(100 * Math.max(...predictions)).toFixed(2)}%</span>
          </div>
        </div>

      </>
    </div>
  );
};

export default HandwritingRecognitionApp;