import Header from "./components/Layout/Header";

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState("closed");

  const handleRefresh = async () => {
   
    setLastRefresh(new Date());
    
  };

  return (
    <ThemeProvider theme={theme}>
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
