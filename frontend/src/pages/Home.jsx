import React, { useState } from 'react';
import axios from 'axios';
import { toast, ToastContainer } from 'react-toastify';
import Navbar from '../components/Navbar';
import illustration from '../assets/illustration.png';

const Home = () => {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/')) {
      toast.error('Only image files are allowed.');
      setFile(null);
      setImagePreview(null);
      setError('Only image files are allowed.');
      return;
    }

    setFile(selectedFile);
    setImagePreview(URL.createObjectURL(selectedFile));
    setError(null);
    setPrediction(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please upload an image file.');
      toast.error('Please upload an image file.');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    const loadingToastId = toast.loading('Text Extraction is in progress...');

    try {
      const apiUrl = import.meta.env.VITE_API_URL;
      const response = await axios.post(`${apiUrl}/predict`, formData);

      setError(null);
      setPrediction(response.data.prediction);

      toast.update(loadingToastId, {
        render: 'Text Extraction completed!',
        type: "success",
        isLoading: false,
        autoClose: 3000,
      });
    } catch (error) {
      let errorMessage = "An unexpected error occurred";
      if (error.response && error.response.data instanceof Blob) {
        const text = await error.response.data.text();
        const json = JSON.parse(text);
        errorMessage = json.error || errorMessage;
      }
      setError(errorMessage);
      setPrediction(null);
      toast.update(loadingToastId, {
        render: errorMessage,
        type: "error",
        isLoading: false,
        autoClose: 3000
      });
    }
  };

  return (
    <>
      <Navbar />
      <div className="min-h-[calc(100vh-96px)] bg-gray-100 bg-[radial-gradient(#d1d5db_1px,transparent_1px)] [background-size:20px_20px] flex flex-col-reverse md:flex-row items-center justify-evenly md:p-4 pt-10 p-4">

        <img
          src={imagePreview || illustration}
          alt="preview"
          className="size-[35rem] object-contain md:block"
        />

        <div className="bg-white p-6 rounded-2xl shadow-md w-full max-w-md">
          <h1 className="text-2xl font-bold mb-4 text-center">Handwriting Recognition</h1>
          <form onSubmit={handleSubmit} className="space-y-4">
            <label className="block">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <div className="cursor-pointer w-full bg-gray-100 border border-gray-300 p-2 rounded text-center hover:bg-gray-200 transition">
                {file ? `Uploaded: ${file.name}` : 'Choose Image File'}
              </div>
            </label>

            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-xl"
            >
              Extract Text
            </button>

            {error && (
              <div className="mt-2 text-center text-red-600">
                {error}
              </div>
            )}

            {prediction && (
              <div className="mt-4 text-center text-green-700 font-medium">
                <strong>Prediction:</strong> {prediction}
              </div>
            )}
          </form>
        </div>

        <ToastContainer />
      </div>
    </>
  );
};

export default Home;
