const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const scripts = ['dislocation/index.html', 'public/dislocation/index.html'].map(path =>
  fs.readFileSync(path, 'utf8').split('<script>')[1].split('</script>')[0]
);
assert.equal(scripts[0], scripts[1], 'Both published dashboard scripts must stay in sync');

for (const script of scripts) {
  const elements = new Map();
  const context = vm.createContext({document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {textContent: '', className: '', style: {}});
      return elements.get(id);
    },
  }});
  // Evaluate render functions without initiating a network request.
  vm.runInContext(script.split('  load().catch')[0], context);
  const render = data => {
    context.data = data;
    vm.runInContext('setStatus(data)', context);
    return elements.get('statusBadge').textContent;
  };
  assert.equal(render({status: 'data_stale', dislocation: true, watch: true, signals_triggered_count: 3}), 'DATA STALE');
  assert.equal(render({status: 'stress_building', dislocation: false, signals_triggered_count: 2}), 'STRESS BUILDING');
  assert.equal(render({status: 'dislocation', dislocation: true, signals_triggered_count: 2, persistence_applied: true}), 'DISLOCATION');
  assert.match(elements.get('statusExplain').textContent, /previously confirmed/);
  assert.equal(render({status: 'normal', signals_triggered_count: 0}), 'NORMAL');
}
console.log('Dashboard regression checks passed for both published copies');
