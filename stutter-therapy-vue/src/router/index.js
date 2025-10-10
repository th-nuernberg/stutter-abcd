import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', component: StartingPage },
  { path: '/kids', component: KidsView },
  { path: '/adults', component: AdultsView },
  { path: '/stutter-types', components: StutterTypes}
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
