#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>


struct CSRGraph
{

    int nodes;

    int edges;


    std::vector<int> row_offsets;

    std::vector<int> column_indices;



    void loadFromEdgeList(const std::string& filename)
    {

        std::ifstream file(filename);


        if(!file.is_open())
        {
            throw std::runtime_error(
                "Could not open graph file"
            );
        }



        std::vector<std::pair<int,int>> edges;


        int src;
        int dst;


        int max_node = 0;



        while(file >> src >> dst)
        {

            edges.push_back({src,dst});

            max_node = std::max(
                max_node,
                std::max(src,dst)
            );

        }



        nodes = max_node + 1;

        this->edges = edges.size();



        row_offsets.assign(
            nodes + 1,
            0
        );



        for(auto& edge : edges)
        {
            row_offsets[
                edge.first + 1
            ]++;
        }



        for(int i=0;i<nodes;i++)
        {
            row_offsets[i+1] += row_offsets[i];
        }



        column_indices.resize(
            this->edges
        );



        std::vector<int> position =
            row_offsets;



        for(auto& edge : edges)
        {

            int src = edge.first;

            column_indices[
                position[src]++
            ] = edge.second;

        }


        std::cout
            << "Loaded graph\n"
            << "Nodes: "
            << nodes
            << "\nEdges: "
            << this->edges
            << "\n";

    }

};
