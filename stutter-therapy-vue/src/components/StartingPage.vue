<template>
  <div class= "header">
    <h1>Stottertherapie</h1>
    <h2>Wähle deine Version aus</h2>
  </div>
  <div class="versions">
    <div class="kids-version">
      <h3>Kinder</h3>
      <p>Beschreibung der Version für Kinder</p>
    </div>
    <div class="adults-version">
      <h3>Erwachsene</h3>
      <p>Beschreibung der Version für</p>
    </div>

  </div>
</template>

<script>
export default {
  name: 'StartingPage' 
}
</script>

<style>
* {
  background-color:ghostwhite
}

.header{
  color: #0057a3;;
}

h1 {
  text-align: center;
  font-size: 45px;
  margin-top: 50px;
}

h2 {
  text-align: center;
  font-size: 25px;
  margin-top: 20px;
  color:black
}

.versions {
  display: flex;
  justify-content: center;
  margin-top: 50px;
  gap: 90px;
}

.kids-version, .adults-version {
  border: 2px solid #ccc;
  border-radius: 2px;
  padding: 20px;
  width: 300px;
  height: 200px;
  text-align: center;
  cursor: pointer;
  transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s;
}

.kids-version:hover, .adults-version:hover {
  transform: translateY(-6px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  border-color: #0057a3;
}
</style>
