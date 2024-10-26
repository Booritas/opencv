/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef CIRCLESGRID_HPP_
#define CIRCLESGRID_HPP_

#include <fstream>
#include <set>
#include <list>
#include <numeric>
#include <map>

class CirclesGridClusterFinder
{
    CirclesGridClusterFinder& operator=(const CirclesGridClusterFinder&);
    CirclesGridClusterFinder(const CirclesGridClusterFinder&);
public:
  CirclesGridClusterFinder(const ncvslideio::CirclesGridFinderParameters &parameters)
  {
    isAsymmetricGrid = parameters.gridType == ncvslideio::CirclesGridFinderParameters::ASYMMETRIC_GRID;
    squareSize = parameters.squareSize;
    maxRectifiedDistance = parameters.maxRectifiedDistance;
  }
  void findGrid(const std::vector<ncvslideio::Point2f> &points, ncvslideio::Size patternSize, std::vector<ncvslideio::Point2f>& centers);

  //cluster 2d points by geometric coordinates
  void hierarchicalClustering(const std::vector<ncvslideio::Point2f> &points, const ncvslideio::Size &patternSize, std::vector<ncvslideio::Point2f> &patternPoints);
private:
  void findCorners(const std::vector<ncvslideio::Point2f> &hull2f, std::vector<ncvslideio::Point2f> &corners);
  void findOutsideCorners(const std::vector<ncvslideio::Point2f> &corners, std::vector<ncvslideio::Point2f> &outsideCorners);
  void getSortedCorners(const std::vector<ncvslideio::Point2f> &hull2f, const std::vector<ncvslideio::Point2f> &patternPoints, const std::vector<ncvslideio::Point2f> &corners, const std::vector<ncvslideio::Point2f> &outsideCorners, std::vector<ncvslideio::Point2f> &sortedCorners);
  void rectifyPatternPoints(const std::vector<ncvslideio::Point2f> &patternPoints, const std::vector<ncvslideio::Point2f> &sortedCorners, std::vector<ncvslideio::Point2f> &rectifiedPatternPoints);
  void parsePatternPoints(const std::vector<ncvslideio::Point2f> &patternPoints, const std::vector<ncvslideio::Point2f> &rectifiedPatternPoints, std::vector<ncvslideio::Point2f> &centers);

  float squareSize, maxRectifiedDistance;
  bool isAsymmetricGrid;

  ncvslideio::Size patternSize;
};

class Graph
{
public:
  typedef std::set<size_t> Neighbors;
  struct Vertex
  {
    Neighbors neighbors;
  };
  typedef std::map<size_t, Vertex> Vertices;

  Graph(size_t n);
  void addVertex(size_t id);
  void addEdge(size_t id1, size_t id2);
  void removeEdge(size_t id1, size_t id2);
  bool doesVertexExist(size_t id) const;
  bool areVerticesAdjacent(size_t id1, size_t id2) const;
  size_t getVerticesCount() const;
  size_t getDegree(size_t id) const;
  const Neighbors& getNeighbors(size_t id) const;
  void floydWarshall(ncvslideio::Mat &distanceMatrix, int infinity = -1) const;
private:
  Vertices vertices;
};

struct Path
{
  int firstVertex;
  int lastVertex;
  int length;

  std::vector<size_t> vertices;

  Path(int first = -1, int last = -1, int len = -1)
  {
    firstVertex = first;
    lastVertex = last;
    length = len;
  }
};

class CirclesGridFinder
{
public:
  CirclesGridFinder(ncvslideio::Size patternSize, const std::vector<ncvslideio::Point2f> &testKeypoints,
                    const ncvslideio::CirclesGridFinderParameters &parameters = ncvslideio::CirclesGridFinderParameters());
  bool findHoles();
  static ncvslideio::Mat rectifyGrid(ncvslideio::Size detectedGridSize, const std::vector<ncvslideio::Point2f>& centers, const std::vector<
      ncvslideio::Point2f> &keypoint, std::vector<ncvslideio::Point2f> &warpedKeypoints);

  void getHoles(std::vector<ncvslideio::Point2f> &holes) const;
  void getAsymmetricHoles(std::vector<ncvslideio::Point2f> &holes) const;
  ncvslideio::Size getDetectedGridSize() const;

  void drawBasis(const std::vector<ncvslideio::Point2f> &basis, ncvslideio::Point2f origin, ncvslideio::Mat &drawImg) const;
  void drawBasisGraphs(const std::vector<Graph> &basisGraphs, ncvslideio::Mat &drawImg, bool drawEdges = true,
                       bool drawVertices = true) const;
  void drawHoles(const ncvslideio::Mat &srcImage, ncvslideio::Mat &drawImage) const;
private:
  void computeRNG(Graph &rng, std::vector<ncvslideio::Point2f> &vectors, ncvslideio::Mat *drawImage = 0) const;
  void rng2gridGraph(Graph &rng, std::vector<ncvslideio::Point2f> &vectors) const;
  void eraseUsedGraph(std::vector<Graph> &basisGraphs) const;
  void filterOutliersByDensity(const std::vector<ncvslideio::Point2f> &samples, std::vector<ncvslideio::Point2f> &filteredSamples);
  void findBasis(const std::vector<ncvslideio::Point2f> &samples, std::vector<ncvslideio::Point2f> &basis,
                 std::vector<Graph> &basisGraphs);
  void findMCS(const std::vector<ncvslideio::Point2f> &basis, std::vector<Graph> &basisGraphs);
  size_t findLongestPath(std::vector<Graph> &basisGraphs, Path &bestPath);
  float computeGraphConfidence(const std::vector<Graph> &basisGraphs, bool addRow, const std::vector<size_t> &points,
                               const std::vector<size_t> &seeds);
  void addHolesByGraph(const std::vector<Graph> &basisGraphs, bool addRow, ncvslideio::Point2f basisVec);

  size_t findNearestKeypoint(ncvslideio::Point2f pt) const;
  void addPoint(ncvslideio::Point2f pt, std::vector<size_t> &points);
  void findCandidateLine(std::vector<size_t> &line, size_t seedLineIdx, bool addRow, ncvslideio::Point2f basisVec, std::vector<
      size_t> &seeds);
  void findCandidateHoles(std::vector<size_t> &above, std::vector<size_t> &below, bool addRow, ncvslideio::Point2f basisVec,
                          std::vector<size_t> &aboveSeeds, std::vector<size_t> &belowSeeds);
  static bool areCentersNew(const std::vector<size_t> &newCenters, const std::vector<std::vector<size_t> > &holes);
  bool isDetectionCorrect();

  static void insertWinner(float aboveConfidence, float belowConfidence, float minConfidence, bool addRow,
                           const std::vector<size_t> &above, const std::vector<size_t> &below, std::vector<std::vector<
                               size_t> > &holes);

  struct Segment
  {
    ncvslideio::Point2f s;
    ncvslideio::Point2f e;
    Segment(ncvslideio::Point2f _s, ncvslideio::Point2f _e);
  };

  //if endpoint is on a segment then function return false
  static bool areSegmentsIntersecting(Segment seg1, Segment seg2);
  static bool doesIntersectionExist(const std::vector<Segment> &corner, const std::vector<std::vector<Segment> > &segments);
  void getCornerSegments(const std::vector<std::vector<size_t> > &points, std::vector<std::vector<Segment> > &segments,
                         std::vector<ncvslideio::Point> &cornerIndices, std::vector<ncvslideio::Point> &firstSteps,
                         std::vector<ncvslideio::Point> &secondSteps) const;
  size_t getFirstCorner(std::vector<ncvslideio::Point> &largeCornerIndices, std::vector<ncvslideio::Point> &smallCornerIndices,
                        std::vector<ncvslideio::Point> &firstSteps, std::vector<ncvslideio::Point> &secondSteps) const;
  static double getDirection(ncvslideio::Point2f p1, ncvslideio::Point2f p2, ncvslideio::Point2f p3);

  std::vector<ncvslideio::Point2f> keypoints;

  std::vector<std::vector<size_t> > holes;
  std::vector<std::vector<size_t> > holes2;
  std::vector<std::vector<size_t> > *largeHoles;
  std::vector<std::vector<size_t> > *smallHoles;

  const ncvslideio::Size_<size_t> patternSize;
  ncvslideio::CirclesGridFinderParameters parameters;
  bool rotatedGrid = false;

  CirclesGridFinder& operator=(const CirclesGridFinder&);
  CirclesGridFinder(const CirclesGridFinder&);
};

#endif /* CIRCLESGRID_HPP_ */
