#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stack>
using namespace std;
const char * cityPath = "city.txt";
const char * distPath = "data.txt";
int dist[20][20];

string city[20] = {"Arad","Bucharest","Craiova","Dobreta","Eforie","Fagaras","Giurgiu","Hirsova","Iasi","Lugoj","Mehadia","Neamt","Oradea","Pitesti","Rimnicu Vilcea","Sibiu","Timisoara","Urziceni","Vaslui","Zerind"};

//下面这个函数配合着vector就会报错，无奈之下只好将城市名列举出来
//void readCity(const char * filepath){
//    ifstream file(cityPath);
//    string temp;
//    int i = 0;
//    if(!file) cout<<"openfailed";
//    while(file>>temp) {
//        // city.push_back(temp);
//        city[i] = temp;
//        i ++;
//    }
//    file.close();
//}
void readDist(const char * filepath) {
    ifstream file(filepath);
    if(!file) {
        cout<<"open file failed"<<endl;
        exit(-1)
    }
    for(int i = 0; i < 20; ++ i) {
        for(int j = 0; j < 20; ++ j) {
            if(i==j) dist[i][j] = 0;
            else dist[i][j] = -1;
        }
    }
    int a,b,c;
    while(file>>a>>b>>c) {
//        cout<<a<<" "<<b<<" "<<c<<endl;
        dist[a][b] = c;
        dist[b][a] = c;
    }
}

//上面是文件读取的部分
//f = g + h
//g表示起始点到当前的距离
//h表示目的地到当前的直线距离

//这个数据表示各个城市到B的直线距离
int strai_dist_B[20] = {366,0,160,242,161,178,77,151,226,244,241,234,380,98,193,253,329,80,199,374};

struct Node{
    int f;
    int g;
    int h;
    int id; //城市的序号
    int father_id;
    Node() {
        f = 0;
        g = 0;
        h = 0;
        id = -1;
        father_id = -1;
    }
    Node(int g, int h, int id, int father_id){
        this->h = h;
        this->g = g;
        this->id = id;
        this->f = g+h;
        this->father_id = father_id;
    }
};

//采用起始点和终点到B城市的直线距离的差的绝对值用来做H
int getH(int start, int end) {
    return abs(strai_dist_B[start] - strai_dist_B[end]);
}

//这个是排序函数，肯定是先排f较小的
//如果出现f相等了，那么就看他们的h，如果h越小，那么说明离的越近
bool cmp(const Node& node1, const Node &node2) {
    if(node1.f < node2.f) return true;
    else if(node1.f > node2.f) return false;
    else {
        if(node1.g < node2.g) return true;
    }
    return false;
}

//判断这个点是否在某个list里面
bool node_in(int id, vector<Node> List) {
    for(auto it = List.begin(); it != List.end(); ++ it) {
        if(it->id == id) return true;
    }
    return false;
}
Node findNode_byId(vector<Node> list, int id) {
    for(int i = 0; i < list.size(); ++ i) {
        if(id == list[i].id) return list[i];
    }
    return Node(-1,-1,-1,-1);
}


vector<Node> openLists;
vector<Node> closeLists;

bool a_star(int start, int end) {
    if(start < 0 || end < 0 || start > 19 || end > 19) return false;
    Node start_node(0,getH(start,end),start,start);
    openLists.push_back(start_node);
    Node center_node;
    while(1) {
        //第一步，查找openList，找到f最小的点
        //排个序
        sort(openLists.begin(), openLists.end(), cmp);
        //接下来需要围绕着当前节点寻找最小f的点
        //并且将它删除
        center_node = *openLists.begin();
        openLists.erase(openLists.begin());
        closeLists.push_back(center_node);
        int temp_start = center_node.id;
        //................................
        //将一个节点的所有的可到达的领域加入到openLists中
        for(int i = 0; i < 20; ++ i) {
            //如果不能直接到达，或者存在于closeList，就跳过
            if(dist[temp_start][i] == -1 || node_in(i, closeLists)) continue;
            //如果这个点不在openList里面，那么就加进去
            if(!node_in(i, openLists)) {
                Node new_node(dist[temp_start][i]+center_node.g, getH(i,end), i, center_node.id);
                openLists.push_back(new_node);
            }
            //如果在openList中，
            else{
                Node node = findNode_byId(openLists, i);
                //如果从当前点更优的话，那么就改变G值，然后将这个node的father变成当前的点
                //并且重新排openList的顺序
                if(center_node.g + dist[center_node.id][node.id] < node.g) {
                    node.g = center_node.g + dist[center_node.id][node.id];
                    node.father_id = center_node.id;
                    node.f = node.g + node.h;
                    sort(openLists.begin(), openLists.end(), cmp);
                }
            }
        }
        //终点在open里面就可以结束了
        if(node_in(end, openLists)) return true;
        if(openLists.empty()) return false;
    }
}

void guide() {
    cout<<"城市编号及名称"<<endl;
    for(int i = 0; i < 20; ++ i) {
        cout<<i<<" ["<<city[i]<<"]"<<endl;
    }
}

int main() {
    
    readDist(distPath);
    int start,end;
    guide();
    cout<<"请输入要查询的起始点城市和终点城市序号:"<<endl;
    cin>>start>>end;

    if(a_star(start, end)) {
        Node node = findNode_byId(openLists, end);
        cout<<"最短路线长度为: "<<node.f<<endl;
        stack<string> track;
        track.push(city[node.id]);
        while(node.id != start) {
            node = findNode_byId(closeLists, node.father_id);
            track.push(city[node.id]);
        }
        cout<<"行进路线应为: ";
        cout<<track.top();
        track.pop();
        while(!track.empty()) {
            cout<<" -> "<<track.top();
            track.pop();
        }
        cout<<endl;
    }
    else {
        cout<<"search failed!"<<endl;
    }
}
