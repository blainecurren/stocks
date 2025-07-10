import React from "react";
import "./Dashboard.css";

const Dashboard = ({ darkMode }) => {
  return (
    <div className="dashboard-container">
      <div className="dashboard-hero">
        <h1 className="dashboard-title">Market Insights Dashboard</h1>
        <p className="dashboard-subtitle">
          Your comprehensive stock market analysis platform
        </p>
      </div>

      <div className="dashboard-content-grid">
        <div
          className={`dashboard-card ${
            darkMode ? "dark-theme" : "light-theme"
          }`}
        >
          <h3>Portfolio Overview</h3>
          <p>Your portfolio performance will display here</p>
        </div>

        <div
          className={`dashboard-card ${
            darkMode ? "dark-theme" : "light-theme"
          }`}
        >
          <h3>Market Trends</h3>
          <p>Live market trends and analysis</p>
        </div>

        <div
          className={`dashboard-card ${
            darkMode ? "dark-theme" : "light-theme"
          }`}
        >
          <h3>Watchlist</h3>
          <p>Your tracked stocks and alerts</p>
        </div>

        <div
          className={`dashboard-card ${
            darkMode ? "dark-theme" : "light-theme"
          }`}
        >
          <h3>News & Analysis</h3>
          <p>Latest market news and expert insights</p>
        </div>

        <div
          className={`dashboard-card ${
            darkMode ? "dark-theme" : "light-theme"
          }`}
        >
          <h3>Performance Metrics</h3>
          <p>Key performance indicators and analytics</p>
        </div>

        <div
          className={`dashboard-card ${
            darkMode ? "dark-theme" : "light-theme"
          }`}
        >
          <h3>Risk Management</h3>
          <p>Risk assessment and management tools</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
