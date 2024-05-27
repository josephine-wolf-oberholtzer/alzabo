import CheckBoxIcon from '@mui/icons-material/CheckBox';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';
import Autocomplete from '@mui/material/Autocomplete';
import Box from '@mui/material/Box';
import Checkbox from '@mui/material/Checkbox';
// import Slider from '@mui/material/Slider';
import TextField from '@mui/material/TextField';
import * as React from 'react';
import Plot from 'react-plotly.js';
import { connect, DispatchProp } from 'react-redux';

import * as api from './api';
import { useAudio } from './app/hooks';
import { RootState } from './app/store';
import { config } from './config';
import { selectors as corpusSelectors } from './features/corpus/corpusSelectors';
import { listDigests, selectAxis, selectDigests } from './features/corpus/corpusSlice';
import { Feature, ScsynthEntry, SelectedAxes } from './types';

interface StateProps {
  digests: string[];
  entries: { [key: string]: ScsynthEntry[] };
  isLoading: boolean;
  selectedAxes: SelectedAxes;
  selectedDigests: string[];
  selectedEntries: ScsynthEntry[];
}

type Props = StateProps & DispatchProp;

const mapStateToProps = (state: RootState): StateProps => ({
  digests: corpusSelectors.getDigests(state),
  entries: corpusSelectors.getEntries(state),
  isLoading: corpusSelectors.getIsLoading(state),
  selectedAxes: corpusSelectors.getSelectedAxes(state),
  selectedDigests: corpusSelectors.getSelectedDigests(state),
  selectedEntries: corpusSelectors.getSelectedEntries(state),
});

const CorpusExplorer = (props: Props) => {
  const { digests, dispatch, entries, selectedAxes, selectedDigests, selectedEntries } = props;
  React.useEffect(() => {
    dispatch(listDigests());
  }, [dispatch]);
  const play = useAudio();
  const icon = <CheckBoxOutlineBlankIcon fontSize="small" />;
  const checkedIcon = <CheckBoxIcon fontSize="small" />;
  return (
    <>
      <Box sx={{ float: 'left', width: 250 }}>
        <Autocomplete
          disableCloseOnSelect
          getOptionLabel={(option) =>
            option.slice(0, 7) + ' (' + (entries[option] ? entries[option].length : '???') + ')'
          }
          limitTags={3}
          multiple
          onChange={(_event: any, newValue: string[]) => {
            dispatch(selectDigests(newValue));
          }}
          options={digests}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Digests"
            />
          )}
          renderOption={(props, option, { selected }) => (
            <li {...props}>
              <Checkbox
                checked={selected}
                checkedIcon={checkedIcon}
                icon={icon}
                style={{ marginRight: 8 }}
              />
              {option.slice(0, 7) + ' (' + (entries[option] ? entries[option].length : '???') + ')'}
            </li>
          )}
          sx={{ backgroundColor: 'white', mb: 2, width: 250 }}
          value={selectedDigests}
        />
        {/*
        <Slider
          onChange={(_event: any, newValue: number | number[]) => {
            dispatch(selectDurations(newValue as unknown as [number, number]));
          }}
          min={0}
          max={10}
          steps={100}
          sx={{ mb: 2 }}
          value={durations}
        />
        */}
        <Autocomplete
          disableClearable
          onChange={(_event: any, newValue: string | null) => {
            dispatch(selectAxis(['x', (newValue ? newValue : selectedAxes.x) as Feature]));
          }}
          options={config.features}
          renderInput={(params) => (
            <TextField
              {...params}
              label="X-axis"
            />
          )}
          sx={{ backgroundColor: 'white', mb: 2, width: 250 }}
          value={selectedAxes.x}
        />
        <Autocomplete
          disableClearable
          onChange={(_event: any, newValue: string | null) => {
            dispatch(selectAxis(['y', (newValue ? newValue : selectedAxes.y) as Feature]));
          }}
          options={config.features}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Y-axis"
            />
          )}
          sx={{ backgroundColor: 'white', mb: 2, width: 250 }}
          value={selectedAxes.y}
        />
        <Autocomplete
          disableClearable
          onChange={(_event: any, newValue: string | null) => {
            dispatch(selectAxis(['z', (newValue ? newValue : selectedAxes.z) as Feature]));
          }}
          options={config.features}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Z-axis"
            />
          )}
          sx={{ backgroundColor: 'white', mb: 2, width: 250 }}
          value={selectedAxes.z}
        />
      </Box>
      <Box sx={{ height: '100%', left: 0, position: 'fixed', top: 0, width: '100%', zIndex: -1 }}>
        <Plot
          config={{
            displayModeBar: false,
            responsive: true,
          }}
          data={[
            {
              customdata: selectedEntries.map((entry) => [entry.digest, entry.start, entry.count]),
              hovertemplate: `${selectedAxes.x}: %{x}<br>${selectedAxes.y}: %{y}<br>${selectedAxes.z}: %{z}`,
              marker: {
                cmax: selectedDigests.length,
                cmin: 0,
                colorscale: 'Portland',
                color: selectedEntries.map((entry: ScsynthEntry) => selectedDigests.indexOf(entry.digest)),
                line: {
                  color: 'white',
                  width: 1,
                },
                opacity: 0.25,
                size: 6,
                symbol: 'circle',
              },
              mode: 'markers',
              type: 'scatter3d',
              x: selectedEntries.map((entry: ScsynthEntry) => entry.features[selectedAxes.x]),
              y: selectedEntries.map((entry: ScsynthEntry) => entry.features[selectedAxes.y]),
              z: selectedEntries.map((entry: ScsynthEntry) => entry.features[selectedAxes.z]),
            },
          ]}
          layout={{
            autosize: true,
            margin: {
              l: 0,
              r: 0,
              b: 0,
              t: 0,
            },
            scene: {
              xaxis: { title: selectedAxes.x },
              yaxis: { title: selectedAxes.y },
              zaxis: { title: selectedAxes.z },
            },
          }}
          onClick={async (event) => {
            const [digest, start, count] = event.points[0].customdata as unknown as [string, number, number];
            const arrayBuffer = await api.fetchAudio(digest, start, count);
            play(arrayBuffer);
          }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </Box>
    </>
  );
};

export default connect(mapStateToProps)(CorpusExplorer);
