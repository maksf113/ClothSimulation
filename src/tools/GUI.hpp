#pragma once
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "ClothParams.hpp"

class GUI
{
public:
	GUI(GLFWwindow* win);
	~GUI();
	void createFrame(ClothParams& params, bool& simulate);
	void draw();
};

inline GUI::GUI(GLFWwindow* window)
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, false);
	const char* glslVersion = "#version 450 core";
	ImGui_ImplOpenGL3_Init(glslVersion);
}

inline GUI::~GUI()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

inline void GUI::createFrame(ClothParams& params, bool& simulate)
{
	// init
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGuiWindowFlags imguiWindowFlags = 0;
	ImGui::SetNextWindowBgAlpha(0.8f);
	
	// controls
	ImGui::Begin("Cloth Simulation Controls");

	// wind
	ImGui::Text("Wind");
	ImGui::SliderFloat("##Wind", &params.wind.z, -50.0f, 50.0f);

	// damping
	ImGui::Separator();
	ImGui::Text("Damping");
	ImGui::SliderFloat("Damping", &params.damping, 0.0f, 0.1f);

	// start/stop
	if (ImGui::Button("Start/Stop"))
		simulate = !simulate;

	ImGui::End();
}

void GUI::draw()
{
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}