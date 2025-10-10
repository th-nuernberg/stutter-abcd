<template>
  <div class="starting-page">
    <div class="header">
      <h1>Stottertherapie</h1>
      <h2>Wähle deine Version aus</h2>
    </div>

    <div class="versions">
      <div class="fluency-shaping">
        <h3>Fluency Shaping</h3>
      </div>

      <router-link to="/stutter-types" class="stutter-type">
        <h3>Stotterart</h3>
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

.versions {
  display: flex;
  justify-content: center;
  margin-top: 70px;
  gap: 90px;
}

.fluency-shaping,
.stutter-type {
  border: 2px solid #ccc;
  border-radius: 10px;
  padding: 20px;
  width: 300px;
  height: 220px;
  text-align: center;
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-decoration: none; /* wichtig für router-link */
  color: inherit; /* Schriftfarbe bleibt */
}
.fluency-shaping:hover,
.stutter-type:hover {
  transform: translateY(-6px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  border-color: #0057a3;
}
</style>
