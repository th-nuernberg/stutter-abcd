<template>
  <div class="starting-page">
    <div class="header">
      <h1>Stottertherapie</h1>
      <h2>Wähle die Stotterart aus</h2>
    </div>

    <div class="stutter-types-choose">
      <!-- Wiederholungen -->
      <router-link to="/wiederholungen" class="stutter-card">
        <h3>Wiederholungen</h3>
        <p>Laute, Silben oder Wörter werden mehrfach wiederholt.</p>
      </router-link>

      <!-- Teilweise Laute -->
      <router-link to="/prolongation" class="stutter-card">
        <h3>Prolongation</h3>
        <p>Laute werden unvollständig oder abgebrochen ausgesprochen.</p>
      </router-link>

      <!-- Blockaden -->
      <router-link to="/blockaden" class="stutter-card">
        <h3>Blockaden</h3>
        <p>Der Sprechfluss wird kurzzeitig blockiert, bevor der Laut entsteht.</p>
      </router-link>
    </div>
  </div>
</template>

<script>
export default {
  name: 'StartingPage'
};
</script>

<style>
* {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

.header {
  color: #0057a3;
  text-align: center;
}

h1 {
  font-size: 45px;
  margin-top: 50px;
}

h2 {
  font-size: 25px;
  margin-top: 20px;
  color: black;
}

.stutter-types-choose {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 70px;
  gap: 40px;
}

.stutter-card {
  border: 2px solid #ccc;
  border-radius: 10px;
  padding: 25px;
  width: 280px;
  height: 200px;
  text-align: center;
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-decoration: none;
  color: inherit;
}

.stutter-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  border-color: #0057a3;
}
</style>
