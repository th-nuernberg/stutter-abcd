import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'

import App from './App.vue'
import StartingPage from './components/StartingPage.vue'
import KidsView from './components/KidsView.vue'
import StutterTypes from './components/StutterTypes.vue'
import AdultsView from './components/AdultsView.vue'


const router = createRouter({
    history: createWebHistory(),
    routes: [
        { path: '/', component: StartingPage },
        { path: '/kids', component: KidsView },
        { path: '/stutter-types', component: StutterTypes },
        { path: '/adults', component: AdultsView}
    ]
});

const app = createApp(App)
app.use(router);

app.mount('#app')