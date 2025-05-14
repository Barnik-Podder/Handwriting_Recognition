import React from "react";
import { Link } from "react-router-dom";
import pen from "../assets/pen.png";

const Navbar = () => {
  return (
    <nav className="h-24 bg-white shadow-md flex items-center justify-start">
      <Link to="/" className="flex items-center space-x-3">
        <img
          src={pen}
          alt="Logo"
          className="size-24 object-contain"
        />
        <span className="text-3xl font-semibold text-gray-800 whitespace-nowrap">Handwritting Recognition</span>
      </Link>
    </nav>
  );
};

export default Navbar;