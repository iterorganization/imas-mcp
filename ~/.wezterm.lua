local wezterm = require 'wezterm'
local config = wezterm.config_builder()

-- 1. Appearance
config.color_scheme = 'Solarized Light (Gogh)'

-- 2. SSH Domain (The "Multiplexer")
config.ssh_domains = {
  {
    name = 'sdcc-remote',
    remote_address = 'sdcc', -- Matches your .ssh/config Host
    remote_wezterm_path = '~/bin/wezterm',
  },
}

-- 3. Leader Key (CTRL-A) for easy splits
config.leader = { key = 'a', mods = 'CTRL', timeout_milliseconds = 1000 }

-- 4. Keybindings for your requested layout
config.keys = {
  -- Split horizontal (Leader + %)
  { mods = 'LEADER', key = '%', action = wezterm.action.SplitHorizontal { domain = 'CurrentPaneDomain' } },
  -- Toggle Zoom (Leader + z) to focus on code or aider
  { mods = 'LEADER', key = 'z', action = wezterm.action.TogglePaneZoomState },
  -- Move between panes
  { mods = 'LEADER', key = 'h', action = wezterm.action.ActivatePaneDirection 'Left' },
  { mods = 'LEADER', key = 'l', action = wezterm.action.ActivatePaneDirection 'Right' },
}

return config
