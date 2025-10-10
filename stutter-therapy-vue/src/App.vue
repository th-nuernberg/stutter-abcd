<template>
  <div>
    <router-view></router-view>

    <footer class="footer">
      <p>© 2025 Stottertherapie </p>
    </footer>
  </div>
</template>

<style>
html, body {
  margin: 0;
  padding: 0;
}

.footer {
  height: 7%;
  background-color: #0057a3; /* Dunkelblau */
  color: white; /* Textfarbe */
  text-align: center; /* Text zentrieren */
  padding: 20px 0; /* Oben und unten Abstand */
  position: fixed; /* Immer am unteren Rand */
  width: 100%; /* Über die ganze Breite */
  bottom: 0; /* am unteren Rand */
}
</style>
