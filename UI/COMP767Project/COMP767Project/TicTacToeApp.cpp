#include "TicTacToeApp.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


int screenWidth, screenHeight;
bool TicTacToeApp::shouldAdvance = false;
int TicTacToeApp::playerAction = -1;

TicTacToeApp::TicTacToeApp()
{
	initOpenGL();
    initInput();
    initBoard();

    //csvFileLoc = sarsaCSV;
    csvFileLoc = "C:/Users/mahya/Google Drive/Project767/SARSA_Q_VALUES.csv";
    //csvFileLoc = "C:/Users/mahya/Google Drive/Project767/a.csv";

    loadQValues();

    turn = starter;
}

void TicTacToeApp::update()
{
    if(!ended)
    {
        // update if it's the agent's turn or it's the player's turn and he has given a valid input
        if (turn == 1)
        {
            //int action = randomAction();
            int action = greedy();

            int i = action / 3;
            int j = action % 3;

            S(i, j) = 2;

            // check termination
            if (ended = checkWinner())
            {
                winner = turn;
                starter = turn;
            }
            // change turn 
            turn = 1 - turn;
        }
        if (TicTacToeApp::playerAction >= 0 && turn == 0)
        {

            int i = TicTacToeApp::playerAction / 3;
            int j = TicTacToeApp::playerAction % 3;

            if (S(i, j) != 0)
            {
                return;
            }

            S(i, j) = 1;

            // check termination
            if (ended = checkWinner())
            {
                winner = turn;
                starter = turn;
            }
            // change turn 
            turn = 1 - turn;

            //TicTacToeApp::shouldAdvance = false;
            TicTacToeApp::playerAction = -1;
        }
        
        vector<int> A;
        getAvailableActions(A);
        if (A.empty() && !ended)
        {
            // it's a tie
            starter = 1 - starter;
            winner = -1;
            ended = true;
        }
    }
    

    // Reset Simulation
    if (glfwGetKey(this->window, GLFW_KEY_R) == GLFW_PRESS)
    {
        reset();
    }
}

void TicTacToeApp::reset()
{
    S.setZero();
    turn = starter;

    TicTacToeApp::playerAction = -1;
    ended = false;
    winner = -1;
}

void TicTacToeApp::getAvailableActions(vector<int>& A)
{
    A.clear();

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (S(i, j) == 0)
                A.push_back(3 * i + j);
        }
    }
}

int TicTacToeApp::greedy()
{
    vector<int> A;
    getAvailableActions(A);

    float best_Q = -1e6;
    int bestAction = -1;

    int x = stateToDecimal(S);

    for (int a : A)
    {
        int key = getKey(x, a);
        float q = Q[key];
       
        if (q > best_Q)
            bestAction = a;
    }

    return bestAction;
}

int TicTacToeApp::randomAction()
{
    vector<int> A;
    getAvailableActions(A);

    return A[rand() % A.size()];
}

bool TicTacToeApp::checkWinner()
{
    // rows
    bool result =      (S(0, 0) == S(0, 1) && S(0, 0) == S(0, 2) && (!S(0, 0) == 0));
    result = result || (S(1, 0) == S(1, 1) && S(1, 0) == S(1, 2) && (!S(1, 0) == 0));
    result = result || (S(2, 0) == S(2, 1) && S(2, 0) == S(2, 2) && (!S(2, 0) == 0));
    //# columns
    result = result || (S(0, 0) == S(1, 0) && S(0, 0) == S(2, 0) && (!S(0, 0) == 0));
    result = result || (S(0, 1) == S(1, 1) && S(0, 1) == S(2, 1) && (!S(0, 1) == 0));
    result = result || (S(0, 2) == S(1, 2) && S(0, 2) == S(2, 2) && (!S(0, 2) == 0));
    //# diagonal
    result = result || (S(0, 0) == S(1, 1) && S(0, 0) == S(2, 2) && (!S(0, 0) == 0));
    result = result || (S(0, 2) == S(1, 1) && S(0, 2) == S(2, 0) && (!S(0, 2) == 0));
    return result;
}

int TicTacToeApp::stateToDecimal(Eigen::Matrix3i ss)
{
    int id = 0;
    int x = 1;
    for (int i = 0; i < 3; i++)
    {
        int y = 1;
        for (int j = 0; j < 3; j++)
        {
            id += ss(i,j) * x * y;
            y *= 3;
        }                
        x *= 27;
    }
    return id;
}

void TicTacToeApp::display()
{
    glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

    if (screenHeight == 0 || screenWidth == 0)
    {
        return;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);

    glMatrixMode(GL_PROJECTION); // projection matrix defines the properties of the camera that views the objects in the world coordinate frame. Here you typically set the zoom factor, aspect ratio and the near and far clipping planes
    glLoadIdentity(); // replace the current matrix with the identity matrix and starts us a fresh because matrix transforms such as glOrpho and glRotate cumulate, basically puts us at (0, 0, 0)
    glViewport(0, 0, screenWidth, screenHeight); // specifies the part of the window to which OpenGL will draw (in pixels), convert from normalised to pixels
    glMatrixMode(GL_MODELVIEW); // (default matrix mode) modelview matrix defines how your objects are transformed (meaning translation, rotation and scaling) in your world
    glLoadIdentity(); // same as above comment

    drawBoard();

    string text = "Your role: O \t \t Turn: ";
    if (turn == 0)
        text += 'O';
    else
        text += 'X';

    printText(-0.8f, 0.8f, 0.0f, 0.0f, 0.0f, GLUT_BITMAP_TIMES_ROMAN_24, text);

    if (ended)
    {
        string winText;
        if (winner == -1)
        {
            winText = "It's a tie!";
        }
        if (winner == 0)
        {
            winText = "Congratulations! you won!";
        }
        if (winner == 1)
        {
            winText = "You lost! try again?";
        }
        
        printText(-0.25f, 0.9f, 0.0f, 0.0f, 0.0f, GLUT_BITMAP_TIMES_ROMAN_24, winText);
        printText(-0.25f, 0.85f, 0.0f, 0.0f, 0.0f, GLUT_BITMAP_TIMES_ROMAN_24, "Press R to play again.");
    }

    // Swap front and back buffers
    glfwSwapBuffers(window);
}

void TicTacToeApp::initOpenGL()
{
    
	if (glfwInit() == false)
	{
		std::cout << "ERROR::GLFW_INIT_FAILED" << "\n";
		glfwTerminate();
	}
    
    glfwWindowHint(GLFW_RESIZABLE, true);
	glfwWindowHint(GLFW_SAMPLES, 9);

	window = glfwCreateWindow(1000, 1000, "Tic-Tac-Toe (Bayran - Quinn)", NULL, NULL);

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

	if (window == nullptr)
	{
		std::cout << "ERROR::GLFW_WINDOW_INIT_FAILED" << "\n";
		glfwTerminate();
	}

	glfwMakeContextCurrent(window); //IMPORTANT!!

	glewExperimental = GL_TRUE;

	//Error
	if (glewInit() != GLEW_OK)
	{
		std::cout << "ERROR::MAIN.CPP::GLEW_INIT_FAILED" << "\n";
		glfwTerminate();
	}
	
    glfwSetWindowShouldClose(window, false);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_MULTISAMPLE);

    glEnable(GL_LINE_SMOOTH);
}

void TicTacToeApp::initInput()
{
    //glfwSetCursorPosCallback(this->window, cursorPosCallBack);
    glfwSetInputMode(this->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    //glfwSetCursorEnterCallback(this->window, cursorEnterCallBack);
    glfwSetMouseButtonCallback(this->window, mouseButtonCallBack);
    glfwSetInputMode(this->window, GLFW_STICKY_MOUSE_BUTTONS, 1);
    //glfwSetScrollCallback(window, scrollCallBack);

    //SandBoxApp::mSpring = new MouseSpring();

    glfwSetKeyCallback(this->window, key_callback);
}

void TicTacToeApp::initBoard()
{
    // reset board
    S.setZero();
}

void TicTacToeApp::printText(float x, float y, float r, float g, float b, void* font, std::string text)
{
    glColor3f(r, g, b);
    glRasterPos2f(x, y);
    int len = text.length();
    for (int i = 0; i < len; i++) {
        glutBitmapCharacter(font, text[i]);
    }
}

void TicTacToeApp::drawBoard()
{
    glLineWidth(5);
    glBegin(GL_LINES);
 
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
    
    glVertex2f(-0.75f, -0.25f);
    glVertex2f(0.75f, -0.25f);

    glVertex2f(-0.75f, 0.25f);
    glVertex2f(0.75f, 0.25f);

    glVertex2f(0.25f, -0.75f);
    glVertex2f(0.25f, 0.75f);

    glVertex2f(-0.25f, -0.75f);
    glVertex2f(-0.25f, 0.75f);

    glEnd();

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float cy = -0.5f * (float)(i - 1);
            float cx = 0.5f * (float)(j - 1);

            if (S(i, j) == 1)
                drawO(cx, cy, 0.4f);
            
            if (S(i, j) == 2)
                drawX(cx, cy, 0.4f);
        }
    }
}

void TicTacToeApp::drawX(float x, float y, float scale)
{
    glLineWidth(10);
    glBegin(GL_LINES);
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);

    glVertex2f(x - scale / 2.0f, y - scale / 2.0f);
    glVertex2f(x + scale / 2.0f, y + scale / 2.0f);

    glVertex2f(x - scale / 2.0f, y + scale / 2.0f);
    glVertex2f(x + scale / 2.0f, y - scale / 2.0f);

    glEnd();
}

void TicTacToeApp::drawO(float x, float y, float scale)
{
    float r = scale / 2.0f;

    glLineWidth(3);
    glBegin(GL_LINES);
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);

    float lastX = x + r, nowX = 0.0f;
    float lastY = y, nowY = 0.0f;
    for (int i = 10; i <= 360; i += 10)
    {
        float theta = i * 3.142f / 180.0f;
        nowX = x + r * cos(theta);
        nowY = y + r * sin(theta);
        glVertex2f(lastX, lastY);
        glVertex2f(nowX, nowY);

        lastX = nowX;
        lastY = nowY;
    }
    glEnd();
}

void TicTacToeApp::displayWinMessage()
{
}

void TicTacToeApp::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);


}

void TicTacToeApp::mouseButtonCallBack(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            
            double xpos = 0, ypos = 0;
            glfwGetCursorPos(window, &xpos, &ypos);

            //cout << xpos << ", " << ypos << endl;

            float x = (float)(xpos) / screenWidth * 2 - 1;
            float y = (float)(ypos) / screenHeight * 2 - 1;

            //cout << x << ", " << y << endl;

            int action = 0;

            if (x <= -0.75f || x >= 0.75f || y <= -0.75f || y >= 0.75f)
            {
                action = -1;
            }
            else
            {
                int ix = (int)floorf(2.0f * (x + 0.75f));
                int iy = (int)floorf(2.0f * (y + 0.75f));

                action = 3 * iy + ix;   
            }

            TicTacToeApp::playerAction = action;

           // cout << action << endl << endl;
        }
    }

}

bool TicTacToeApp::shouldCloseWindow()
{
	return glfwWindowShouldClose(window);
}

void TicTacToeApp::loadQValues()
{
	Q.clear();

    // File pointer 
    fstream fin;

    // Open an existing file 
    fin.open(csvFileLoc, ios::in);
    
    int count = 0;

    // Read the Data from the file 
    // as String Vector 
    vector<string> row;
    string line, word, temp;

    while (fin >> temp) {

        row.clear();

        // read an entire row and 
        // store it in a string variable 'line' 
        //getline(fin, line);

        // used for breaking words 
        stringstream s(temp);//s(line);
        count++;
        // read every column data of a row and 
        // store it in a string variable, 'word'
        char sep = ',';
        while (getline(s, word, sep)) 
        {

            // add all the column data 
            // of a row to a vector 
            row.push_back(word);
        }
        if (row.size() < 3)
            cout << "File not OK" << endl;
        else
        {
            int key = getKey(stoi(row[0]), stoi(row[1]));
            Q[key] = stof(row[2]);
        }
    }
    if (count == 0)
        cout << "Record not found\n";
   /* else
    {
        for (auto it = Q.begin(); it != Q.end(); ++it)
            cout << " " << it->first << ":" << it->second << endl;
    }*/
    
    fin.close();
}

int TicTacToeApp::getKey(int s, int a)
{
    int largePrime = 100003;
    return s + largePrime * a;
}
