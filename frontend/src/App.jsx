import { useState } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import Header from "./components/header/Header";
import Dashboard from "./components/dashboard/Dashboard";

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState("closed");

  // Create theme based on darkMode state
  const theme = createTheme({
    palette: {
      mode: darkMode ? "dark" : "light",
    },
  });

  const handleRefresh = async () => {
    setLastRefresh(new Date());
    // Add your refresh logic here
  };

  const handleThemeToggle = () => {
    setDarkMode(!darkMode);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Header
        onThemeToggle={handleThemeToggle}
        onRefresh={handleRefresh}
        lastRefreshTime={lastRefresh}
        marketStatus={marketStatus}
      />
      <Dashboard darkMode={darkMode} />
    </ThemeProvider>
  );
}

export default App;
