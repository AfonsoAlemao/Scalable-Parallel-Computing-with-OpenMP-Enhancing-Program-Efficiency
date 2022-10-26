#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }
  
  
  /*
     TODO STUDENTS: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
    double* score_old, global_diff = 0;
    score_old = (double*) malloc(sizeof(double) * numNodes);
    bool converged = false;
    double aux = 0;
    int chunk_size = (numNodes + 80000 - 1) / 80000;
    // double densidade = 2 * num_edges(g) / numNodes;
    while (!converged) {
      global_diff = 0;
      aux = 0;
      #pragma omp parallel shared(aux, numNodes, score_old, solution, global_diff)
      {
        double mydiff = 0;
        double myaux = 0;
        double mysum = 0;

        #pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < numNodes; ++i) {
          score_old[i] = solution[i];
          //solution[i] = 0;
        }

        // compute score_new[vi] for all nodes vi:
        //score_new[vi] = sum over all nodes vj reachable from incoming edges { score_old[vj] / number of edges leaving vj  }
        //if (densidade < 1) {
        # pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < numNodes; ++i) {
          double auxiliar = 0;
          const Vertex* start = incoming_begin(g, i);
          const Vertex* end = incoming_end(g, i);
          for (const Vertex* v = start; v != end; v++) {
            // Edge (i, *v);
            auxiliar += score_old[*v] / outgoing_size(g, *v);
          }
          solution[i] = auxiliar;
          solution[i] = (damping * auxiliar) + (1.0-damping) / numNodes;
          if (outgoing_size(g, i) == 0) {
            myaux += damping * score_old[i] / numNodes;
          }
        }
        /*}
        else {
          for (int i = 0; i < numNodes; ++i) {
            const Vertex* start = incoming_begin(g, i);
            const Vertex* end = incoming_end(g, i);
            int chunk = (num_edges(g) / numNodes) + 1;
            if (chunk < 8) {
              chunk = 16;
            }
            # pragma omp for schedule(dynamic, chunk)
            for (const Vertex* v = start; v != end; v++) {
              // Edge (i, *v);
              solution[i] += score_old[*v] / outgoing_size(g, *v);
            }
            #pragma omp atomic
            solution[i] += mysum;
            #pragma omp barrier
          }
        }
        */

        /*#pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < numNodes; ++i) {
          solution[i] = (damping * solution[i]) + (1.0-damping) / numNodes;
        }*/

        //score_new[vi] += sum over all nodes v in graph with no outgoing edges { damping * score_old[v] / numNodes }
        /*#pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < numNodes; ++i) {
          if (outgoing_size(g, i) == 0) {
            myaux += damping * score_old[i] / numNodes;
          }
        }*/

        #pragma omp atomic
        aux += myaux;
        #pragma omp barrier

        /*#pragma omp for reduction(+:aux)
        for (int i = 0; i < numNodes; ++i) {
          if (outgoing_size(g, i) == 0) {
            aux += damping * score_old[i] / numNodes;
          }
        }*/

        #pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < numNodes; ++i) {
            solution[i] += aux;
        }
      

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        // global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
      
        #pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < numNodes; ++i) {
          double aux1 = solution[i], aux2 = score_old[i];

          if (aux1 > aux2) {
            mydiff += aux1 - aux2;
          }
          else {
            mydiff += aux2 - aux1;
          }
          
        }
      
        #pragma omp atomic
        global_diff += mydiff;

        /*#pragma omp for reduction(+:global_diff)
        for (int i = 0; i < numNodes; ++i) {
          global_diff += abs(solution[i] - score_old[i]);
        }*/
      }

      converged = (global_diff < convergence);
    }
    free(score_old);
}
