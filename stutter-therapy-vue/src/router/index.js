import { createRouter, createWebHistory } from 'vue-router'
import StartingPage from '../components/StartingPage.vue'
import App from '../App.vue' // Zielseite / andere Komponente

const routes = [
  { path: '/', name: 'Home', component: StartingPage },
  { path: '/app', name: 'App', component: App },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
