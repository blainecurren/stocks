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
  alpha,
} from "@mui/material";
import {
  Brightness4 as DarkIcon,
  Brightness7 as LightIcon,
  TrendingUp as TrendingUpIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreIcon,
  Circle as CircleIcon,
} from "@mui/icons-material";

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
    // Handle export functionality
    console.log("Exporting data...");
    handleMenuClose();
  };

  const handleSettings = () => {
    // Handle settings
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
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        backgroundColor:
          theme.palette.mode === "dark"
            ? alpha(theme.palette.background.paper, 0.8)
            : theme.palette.primary.main,
        backdropFilter: "blur(10px)",
        borderBottom: `1px solid ${
          theme.palette.mode === "dark" ? theme.palette.divider : "transparent"
        }`,
      }}
    >
      <Toolbar sx={{ justifyContent: "space-between" }}>
        {/* Left Section - Logo and Title */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <TrendingUpIcon sx={{ fontSize: 28 }} />
            <Typography
              variant="h6"
              component="h1"
              sx={{
                fontWeight: 600,
                display: { xs: "none", sm: "block" },
              }}
            >
              Market Insights
            </Typography>
          </Box>

          {/* Market Status Chip */}
          <Chip
            icon={<CircleIcon sx={{ fontSize: 12 }} />}
            label={getMarketStatusText()}
            color={getMarketStatusColor()}
            size="small"
            sx={{
              fontWeight: 500,
              "& .MuiChip-icon": {
                animation:
                  marketStatus === "open" ? "pulse 2s infinite" : "none",
              },
              "@keyframes pulse": {
                "0%": { opacity: 1 },
                "50%": { opacity: 0.3 },
                "100%": { opacity: 1 },
              },
            }}
          />
        </Box>

        {/* Right Section - Actions */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          {/* Last Refresh Time */}
          <Typography
            variant="caption"
            sx={{
              color:
                theme.palette.mode === "dark" ? "text.secondary" : "inherit",
              display: { xs: "none", md: "block" },
              mr: 1,
            }}
          >
            Updated: {formatRefreshTime(lastRefreshTime)}
          </Typography>

          {/* Refresh Button */}
          <Tooltip title="Refresh data">
            <IconButton
              color="inherit"
              onClick={onRefresh}
              sx={{
                transition: "transform 0.3s",
                "&:active": {
                  transform: "rotate(360deg)",
                },
              }}
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
            <IconButton color="inherit" onClick={handleMenuOpen} sx={{ ml: 1 }}>
              <MoreIcon />
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            anchorOrigin={{
              vertical: "bottom",
              horizontal: "right",
            }}
            transformOrigin={{
              vertical: "top",
              horizontal: "right",
            }}
          >
            <MenuItem onClick={handleExport}>Export Data</MenuItem>
            <MenuItem onClick={handleSettings}>Settings</MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

// Default props
Header.defaultProps = {
  marketStatus: "closed",
  lastRefreshTime: null,
};

export default Header;
