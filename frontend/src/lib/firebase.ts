import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyA1wdCketK8ILBGWUiruLCItZ5oZrjzCWw",
  authDomain: "automatic-champion.firebaseapp.com",
  projectId: "automatic-champion",
  storageBucket: "automatic-champion.firebasestorage.app",
  messagingSenderId: "480728374149",
  appId: "1:480728374149:web:d96d093700cd85bb8642ec",
  measurementId: "G-Z9HMWBMFCJ",
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
