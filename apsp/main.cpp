#include "main.h"
#define INFI 1073741823
using namespace std;
int main(int argc, char* argv[]){
    int a[40005][40005]{INFI};
    int v,e;
    cin>>v>>e;
    int src,dst,w;
    for(int i=0;i<e;i++)
    {
        cin>>src>>dst>>w;
        a[src][dst]=w;
    }
    for(int i=0;i<v;i++)
    {
        a[i][i]=0;
    }



    for(int i=0;i<v;i++)
    {
        for(int j=0;j<v;j++)
        {
            if(j)
                cout<<" ";
            cout<<a[i][j];
        }
        if(i==v-1)
            cout<<endl;
    }
    return 0;
}