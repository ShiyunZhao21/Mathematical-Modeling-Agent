// 封转路由
import { createRouter, createWebHistory } from "vue-router";
// 路由配置
// meau 需要登录后才能访问

const routes = [
	{
		path: "/",
		component: () => import("@/pages/index.vue"),
	},
	{
		path: "/login",
		component: () => import("@/pages/login/index.vue"),
	},
	{
		path: "/chat",
		component: () => import("@/pages/chat/index.vue"),
	},
	{
		path: "/task/:task_id",
		component: () => import("@/pages/task/index.vue"),
		props: true,
	},
];

// 创建路由
const router = createRouter({
	history: createWebHistory(),
	routes,
});

// 路由守卫
// router.beforeEach((to, from, next) => {})

export default router;
