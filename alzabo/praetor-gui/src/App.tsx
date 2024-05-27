import GraphicEqSharpIcon from '@mui/icons-material/GraphicEqSharp';
import AppBar from '@mui/material/AppBar';
import Container from '@mui/material/Container';
import Divider from '@mui/material/Divider';
import IconButton from '@mui/material/IconButton';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import * as React from 'react';

import CorpusExplorer from './CorpusExplorer';

const App = () => {
  return (
    <Container
      maxWidth={false}
      sx={{ my: 3 }}
    >
      <AppBar
        position="static"
        sx={{ py: 1, mb: 3 }}
      >
        <Toolbar variant="dense">
          <IconButton
            aria-label="menu"
            color="inherit"
            edge="start"
            sx={{ mr: 1 }}
          >
            <GraphicEqSharpIcon />
          </IconButton>
          <Typography
            color="inherit"
            component="div"
            sx={{ mr: 2 }}
            variant="h6"
          >
            praetor
          </Typography>
          <Divider
            flexItem
            orientation="vertical"
            sx={{ backgroundColor: 'white', display: 'inline' }}
            variant="middle"
          />
          <Typography
            color="inherit"
            component="div"
            sx={{ fontWeight: 100, ml: 2 }}
            variant="h6"
          >
            corpus explorer
          </Typography>
        </Toolbar>
      </AppBar>
      <CorpusExplorer />
    </Container>
  );
};

export default App;
