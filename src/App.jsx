import { useState } from "react";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import Header from "./components/Layout/Header";

// Create a basic theme
const theme = createTheme({
  palette: {
    mode: "dark",
  },
});

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState("closed");

  const handleRefresh = async () => {
    setLastRefresh(new Date());
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Header
        onThemeToggle={() => setDarkMode(!darkMode)}
        onRefresh={handleRefresh}
        lastRefreshTime={lastRefresh}
        marketStatus={marketStatus}
      />
      {/* Rest of your app */}
    </ThemeProvider>
  );
}

export default App;
