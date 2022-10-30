#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

//#define VERBOSE
void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    # pragma omp parallel for schedule(dynamic, (frontier->count + 24 - 1) / 24)
    for (int i = 0; i < frontier->count; i++) {
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            int index = 0;
           
            if (distances[outgoing] == NOT_VISITED_MARKER) {
                
                distances[outgoing] = distances[node] + 1;    
                
                //__sync_bool_compare_and_swap (&index, index, new_frontier->count++);
                # pragma omp critical 
                {
                index = new_frontier->count++;
                }

                new_frontier->vertices[index] = outgoing;
            }
            
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for schedule(dynamic, 100)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(list1.vertices);
    free(list2.vertices);
}

/**************************************************************************************
 * ProcuraBinaria ()
 *
 * Argumentos: a - tabela de items onde sera efetuada a procura
 *             l, r - limites da tabela
 *             x - elemento a pesquisar
 * 			   less, equal - funcoes com comparacoes a ser feitas para encontrar x em a
 * Retorna: inteiro com o indice onde x se encontra na tabela a,
 * 			ou -1 se nao for encontrado
 * Efeitos Secundarios: nenhum
 *
 * Descricao: caso encontre o elemento na tabela retorna indice, caso contrario -1
 * 			  O(lgN)
 **************************************************************************************/

int ProcuraBinaria (int a[], int l, int r, int x) {
    while (r >= l) {
		int m = (l + r) / 2;
		if (x == a[m]) return m;
		if (x < a[m]) r = m - 1;
		else l = m + 1;
	}
	return -1;
}


// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    /*for each vertex v in graph:
        if v has not been visited AND v shares an incoming edge with a vertex u on the frontier:
            add vertex v to frontier;*/
    if (frontier->count == 0) {
        return;
    }
    int dist_frontier = distances[frontier->vertices[0]];
    // printf("dist_frontier = %d\n", dist_frontier);

    # pragma omp parallel for schedule(dynamic, (frontier->count + 24 - 1) / 24)
    for (int i = 0; i < num_nodes(g); i++) {
        // printf("Tou no vertice %d\n", i);
        int j = 0;
        if (distances[i] == NOT_VISITED_MARKER) {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                int index = 0;
                // printf("Vizinho %d com distance = %d\n", incoming, distances[incoming]);
                if(distances[incoming] == dist_frontier) {
                    int node = frontier->vertices[j];

                    // printf("Adicionei ligacao %d-%d\n", i, incoming);
                    distances[i] = dist_frontier + 1;    
                    
                    # pragma omp critical 
                    {
                        index += new_frontier->count++;
                    }

                    new_frontier->vertices[index] = i;

                    break;
                }
            }
        }
    }
    // printf("\n\n");
}

    /*for (int i = 0; i < num_nodes(g); i++) {
        printf("%d\n", i);
        int j = 0;
        // printf("Tou no vertice %d\n", i);

        if (distances[i] == NOT_VISITED_MARKER) {
            //printf("Vertice %d ainda nao foi visitado\n", i);
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];

            // printf("Lista de vizinhos\n");
            // for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            //     printf("%d\n", g->incoming_edges[neighbor]);
            // }

            for (j = 0; j < frontier->count; j++) {
                int node = frontier->vertices[j];
                // printf("Vou procurar se tenho um vizinho igual ao vertice %d que pertence à fronteira\n", node);
                int neighbor = ProcuraBinaria(g->incoming_edges, start_edge, end_edge - 1, node);
                
                if (neighbor != -1) {
                    // printf("Encontrei um vizinho %d que está na fronteira\n", neighbor);
                    int incoming = g->incoming_edges[neighbor];
                    int index = 0;
                
                    // printf("Adicionei ligacao %d-%d\n", i, incoming);
                    distances[i] = distances[incoming] + 1;    
                    
                    # pragma omp critical 
                    {
                        index += new_frontier->count++;
                    }

                    new_frontier->vertices[index] = i;

                    break;
                } 
                // else {
                //    printf("Nao encontrei um vizinho que está na fronteira\n");
                //}
            }

        }
    }*/
    

     
    
    /*for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            int index = 0;
            {
            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;    
                
                index += new_frontier->count++;

                new_frontier->vertices[index] = outgoing;
            }
            }
        }
    }*/
   

    
        /*if (distances[i] == NOT_VISITED_MARKER) {
            // printf("i = %d\n", i);
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[i + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                int index = 0;

                if(distances[incoming] != NOT_VISITED_MARKER) {
                    for (j = 0; j < frontier->count; j++) {
                        int node = frontier->vertices[j];

                        if (node == incoming) {
                            // printf("Adicionei ligacao %d-%d\n", i, incoming);
                            distances[i] = distances[incoming] + 1;    
                            
                            # pragma omp critical 
                            {
                                index += new_frontier->count++;
                            }

                            new_frontier->vertices[index] = i;

                            j = frontier->count + 2;
                        }
                    }
                }

                if (j == frontier->count + 2) {
                    break;
                }
            }
        }*/

void bfs_bottom_up(Graph graph, solution* sol)
{
    // TODO STUDENTS:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for schedule(dynamic, 100)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        //vertex_set* tmp;
        //__sync_bool_compare_and_swap ((vertex_set*) tmp, (vertex_set*) frontier, (vertex_set*) new_frontier);

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(list1.vertices);
    free(list2.vertices);
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // TODO STUDENTS:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    # pragma omp parallel for schedule(dynamic, 100)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        // Top down complexity: number of frontier nodes
        // Bottom up complexity: number of nodes outside the bfs ~ number of nodes - number of frontier nodes
        if (frontier->count * frontier->count > graph->num_nodes) {
            bottom_up_step(graph, frontier, new_frontier, sol->distances);
        }
        else {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        //vertex_set* tmp;
        //__sync_bool_compare_and_swap ((vertex_set*) tmp, (vertex_set*) frontier, (vertex_set*) new_frontier);

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(list1.vertices);
    free(list2.vertices);
}
