import React, { useState } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Tooltip,
  Menu,
  MenuItem,
  Chip,
  useTheme,
} from "@mui/material";
import {
  Brightness4 as DarkIcon,
  Brightness7 as LightIcon,
  TrendingUp as TrendingUpIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreIcon,
  Circle as CircleIcon,
} from "@mui/icons-material";
import "./Header.css";

const Header = ({
  onThemeToggle,
  onRefresh,
  lastRefreshTime,
  marketStatus,
}) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = useState(null);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleExport = () => {
    console.log("Exporting data...");
    handleMenuClose();
  };

  const handleSettings = () => {
    console.log("Opening settings...");
    handleMenuClose();
  };

  // Format last refresh time
  const formatRefreshTime = (time) => {
    if (!time) return "Never";
    const now = new Date();
    const refreshTime = new Date(time);
    const diffMinutes = Math.floor((now - refreshTime) / 60000);

    if (diffMinutes < 1) return "Just now";
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffMinutes < 1440) return `${Math.floor(diffMinutes / 60)}h ago`;
    return refreshTime.toLocaleDateString();
  };

  // Determine market status color
  const getMarketStatusColor = () => {
    switch (marketStatus) {
      case "open":
        return "success";
      case "closed":
        return "error";
      case "pre-market":
      case "after-hours":
        return "warning";
      default:
        return "default";
    }
  };

  const getMarketStatusText = () => {
    switch (marketStatus) {
      case "open":
        return "Market Open";
      case "closed":
        return "Market Closed";
      case "pre-market":
        return "Pre-Market";
      case "after-hours":
        return "After Hours";
      default:
        return "Unknown";
    }
  };

  return (
    <>
      <AppBar
        elevation={0}
        className={`header-appbar ${
          theme.palette.mode === "dark" ? "dark-mode" : "light-mode"
        }`}
      >
        <Toolbar className="header-toolbar">
          {/* Left Section - Logo and Title */}
          <Box className="header-left">
            <Box className="header-logo">
              <TrendingUpIcon className="header-logo-icon" />
              <Typography variant="h6" component="h1" className="header-title">
                Market Insights
              </Typography>
            </Box>

            {/* Market Status Chip */}
            <Chip
              icon={<CircleIcon className="market-status-icon" />}
              label={getMarketStatusText()}
              color={getMarketStatusColor()}
              size="small"
              className={`market-status-chip ${
                marketStatus === "open" ? "market-open" : ""
              }`}
            />
          </Box>

          {/* Right Section - Actions */}
          <Box className="header-right">
            {/* Last Refresh Time */}
            <Typography variant="caption" className="header-refresh-time">
              Updated: {formatRefreshTime(lastRefreshTime)}
            </Typography>

            {/* Refresh Button */}
            <Tooltip title="Refresh data">
              <IconButton
                color="inherit"
                onClick={onRefresh}
                className="header-refresh-btn"
              >
                <RefreshIcon />
              </IconButton>
            </Tooltip>

            {/* Theme Toggle */}
            <Tooltip
              title={`Switch to ${
                theme.palette.mode === "dark" ? "light" : "dark"
              } mode`}
            >
              <IconButton color="inherit" onClick={onThemeToggle}>
                {theme.palette.mode === "dark" ? <LightIcon /> : <DarkIcon />}
              </IconButton>
            </Tooltip>

            {/* More Options Menu */}
            <Tooltip title="More options">
              <IconButton
                color="inherit"
                onClick={handleMenuOpen}
                className="header-more-btn"
              >
                <MoreIcon />
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
              className="header-menu"
              anchorOrigin={{
                vertical: "bottom",
                horizontal: "right",
              }}
              transformOrigin={{
                vertical: "top",
                horizontal: "right",
              }}
            >
              <MenuItem onClick={handleExport} className="header-menu-item">
                Export Data
              </MenuItem>
              <MenuItem onClick={handleSettings} className="header-menu-item">
                Settings
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>
    </>
  );
};

// Default props
Header.defaultProps = {
  marketStatus: "closed",
  lastRefreshTime: null,
};

export default Header;
