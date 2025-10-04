<template>
  <div id="app">

    <StartingPage />
  </div>
</template>

<script>
import StartingPage from './components/StartingPage.vue' // Importieren

export default {
  name: 'App',
  components: {
    StartingPage // Registrieren
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
